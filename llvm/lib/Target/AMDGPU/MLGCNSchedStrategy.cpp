//===-- MLGCNSchedStrategy.cpp - GCN ML Scheduler Strategy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
//===----------------------------------------------------------------------===//

#include "MLGCNSchedStrategy.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include <cstdint>

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<MLGCNSchedStrategy::SchedMode>
    EnabledMode("sched-mode", cl::Hidden,
                cl::init(MLGCNSchedStrategy::SchedMode::Default),
                cl::desc("Set ML Sched Mode"),
                cl::values(clEnumValN(MLGCNSchedStrategy::SchedMode::Default,
                                      "default", "Default"),
                           clEnumValN(MLGCNSchedStrategy::SchedMode::Release,
                                      "release", "precompiled"),
                           clEnumValN(MLGCNSchedStrategy::SchedMode::Development,
                                      "development", "for training")));

static cl::opt<std::string> MLSchedTrainingLog(
    "mlsched-training-log", cl::Hidden,
    cl::desc("Training log for the instruction scheduling model"));

static cl::opt<std::string> MLModelUnderTraining(
    "mlsched-model", cl::Hidden,
    cl::desc("The model being trained for instruction scheduling"));

#define SchedDecisionName "index_to_sched"
static const TensorSpec DecisionSpec =
    TensorSpec::createSpec<int64_t>(SchedDecisionName, {1});
static const TensorSpec Reward = TensorSpec::createSpec<float>("reward", {1});

#define _DECL_FEATURES(type, name, shape, _)                                   \
  TensorSpec::createSpec<type>(#name, shape),
#define _DECL_TRAIN_FEATURES(type, name, shape, _)                             \
  TensorSpec::createSpec<type>(std::string("action_") + #name, shape),

// --------------
// Features table
// --------------
static const int MaxNumInstCandidates = 64 * 2;
static const std::vector<int64_t> PerInstrFeatureShape{1, MaxNumInstCandidates};
#define SCHED_FEATURES_LIST(M)					\
  M(int64_t, mask, PerInstrFeatureShape,			\
    "boolean values, 1 if the position is valid")		\
  M(int64_t, is_top, PerInstrFeatureShape,			\
    "boolean values, 1 if the instruction is from Top Queue")	\
  M(int64_t, is_bot, PerInstrFeatureShape,			\
    "boolean values, 1 if the instruction is from Bot Queue")	\
  M(int64_t, excess, PerInstrFeatureShape, "")			\
  M(int64_t, current_max, PerInstrFeatureShape, "")		\
  M(int64_t, critical_max, PerInstrFeatureShape, "")		\
  M(int64_t, su_latency, PerInstrFeatureShape, "")		\
  M(int64_t, su_height, PerInstrFeatureShape, "")		\
  M(int64_t, su_depth, PerInstrFeatureShape, "")		\
  M(int64_t, su_succs_left, PerInstrFeatureShape, "")		\
  M(int64_t, su_preds_left, PerInstrFeatureShape, "")		\
  M(int64_t, su_succs, PerInstrFeatureShape, "")		\
  M(int64_t, su_preds, PerInstrFeatureShape, "")		\
  M(int64_t, sgpr_critical_limit, {1}, "")			\
  M(int64_t, vgpr_critical_limit, {1}, "")			\
  M(int64_t, sgpr_excess_limit, {1}, "")			\
  M(int64_t, vgpr_excess_limit, {1}, "")

// Named features index.
enum SchedFeatureIDs {
#define _FEATURE_IDX_SIMPLE(_, name, __, ___) name
#define _FEATURE_IDX(A, B, C, D) _FEATURE_IDX_SIMPLE(A, B, C, D),
  SCHED_FEATURES_LIST(_FEATURE_IDX) FeatureCount,
#undef _FEATURE_IDX
#undef _FEATURE_IDX_SIMPLE
};


template <typename T> static size_t getTotalSize(const std::vector<int64_t> &Shape) {
  size_t Ret = sizeof(T);
  for (const auto V : Shape)
    Ret *= V;
  return Ret;
}

static void resetRunnerInput(MLModelRunner &Runner) {
  LLVM_DEBUG(dbgs() << "[GCNSchedStrategy::resetRunnerInput]\n");
#define _RESET(TYPE, NAME, SHAPE, __)                            \
  std::memset(Runner.getTensorUntyped(SchedFeatureIDs::NAME), 0, \
              getTotalSize<Type>(SHAPE));
  SCHED_FEATURES_LIST(_RESET)
#undef _RESET
}

MLGCNSchedStrategy::MLGCNSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C), Mode(EnabledMode) {
  if (MLModelUnderTraining.empty() && MLSchedTrainingLog.empty()) {
    Log = nullptr;
    Runner = nullptr;
    LLVM_DEBUG(dbgs() << "[MLGCNSchedStrategy] In development mode, logging or "
	       << "a training model shuold be provided.");
    return;
  }

  std::vector<TensorSpec> InputFeatures = {SCHED_FEATURES_LIST(_DECL_FEATURES)};
  std::vector<TensorSpec> TrainingInputFeatures = {
    SCHED_FEATURES_LIST(_DECL_TRAIN_FEATURES)
    TensorSpec::createSpec<float>("action_discount", {1}),
    TensorSpec::createSpec<int32_t>("action_step_type", {1}),
    TensorSpec::createSpec<float>("action_reward", {1})};

  // Mode == SchedMode::Development
  LLVMContext &Ctx = C->MF->getFunction().getContext();
  if (MLModelUnderTraining.empty())
    Runner = std::make_unique<NoInferenceModelRunner>(Ctx, InputFeatures);
  else
    Runner = ModelUnderTrainingRunner::createAndEnsureValid(
        Ctx, MLModelUnderTraining, SchedDecisionName, TrainingInputFeatures);
  if (!Runner) {
    Ctx.emitError("MLGCNSchedStrategy: could not set up the model runner");
    return;
  }
  if (MLSchedTrainingLog.empty())
    return;

  std::error_code EC;
  auto OS = std::make_unique<raw_fd_ostream>(MLSchedTrainingLog, EC);
  if (EC) {
    Ctx.emitError("[MLGCNSchedStrategy] " + EC.message() + ":" + MLSchedTrainingLog);
    return;
  }

  std::vector<TensorSpec> LFS = InputFeatures;
  if (auto *MUTR = dyn_cast<ModelUnderTrainingRunner>(Runner.get()))
    append_range(LFS, MUTR->extraOutputsForLoggingSpecs());
  LFS.push_back(DecisionSpec);

  Log = std::make_unique<Logger>(std::move(OS), LFS, Reward,
				 /*IncludeReward*/ true);  
}


SUnit *MLGCNSchedStrategy::pickNode(bool &IsTopNode) {
    if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }

  if (Log != nullptr)
    resetRunnerInput(*Runner);

  SUnit *SU;
  int64_t SchedIndex = -1;

  CandPolicy BotPolicy;
  setPolicy(BotPolicy, false, Bot, &Top);
  CandPolicy TopPolicy;
  setPolicy(TopPolicy, false, Top, &Bot);
  extractFeatures(Top, TopPolicy, DAG->getTopRPTracker(), /*IsBottomUp=*/false);
  extractFeatures(Bot, BotPolicy, DAG->getBotRPTracker(), /*IsBottomUp=*/true);
  extractGlobalFeatures();

  // make decision here
  if (isa<ModelUnderTrainingRunner>(getRunner())) {
    SU = pickNodeByModel();
    // TODO: skip do-while if no need. // how to determine it?
    // when to keep collecting data?
  }

  do {
    if (RegionPolicy.OnlyTopDown) {
      SU = Top.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, NoPolicy, DAG->getTopRPTracker(), TopCand,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
	SchedIndex = Top.Available.getPos(SU);
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = Bot.pickOnlyChoice();
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, NoPolicy, DAG->getBotRPTracker(), BotCand,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
	SchedIndex = Bot.Available.getPos(SU);
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode, SchedIndex);
    }
  } while (SU->isScheduled);

  if (Log != nullptr)
    logMLFeatures(SchedIndex, IsTopNode);

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());
  return SU;
}

SUnit *MLGCNSchedStrategy::pickNodeByModel() {
  int64_t Idx = Runner->evaluate<int64_t>();

  if (Idx % 2) {
    --Idx;
    ReadyQueue &TopQ = Top.Available;
    return TopQ.elements()[Idx];
  } else {
    ReadyQueue &BotQ = Bot.Available;
    return BotQ.elements()[Idx];
  }
}

void MLGCNSchedStrategy::logMLFeatures(int64_t SchedIndex, bool &IsTopNode) {
  //  if (Log->hasObservationInProgress())
  //    Log->logReward<float>(0.0);
  LLVM_DEBUG(dbgs() << "[MLGCNSchedStrategy::logMLFeatures] schedule index: "
	     << SchedIndex << "\n");
  if (SchedIndex < 0) {
    return;
  }

  Log->startObservation();
  size_t CurrentFeature = 0;
  for (; CurrentFeature < SchedFeatureIDs::FeatureCount; ++CurrentFeature) {
    Log->logTensorValue(CurrentFeature,
                        reinterpret_cast<const char *>(getRunner().getTensorUntyped(CurrentFeature)));
  }
  // Log the decision (index of ready queue) // may need to add direction (top/bottom)
  SchedIndex = SchedIndex * 2 + IsTopNode;
  Log->logTensorValue(CurrentFeature, reinterpret_cast<const char *>(&SchedIndex));
  Log->endObservation();

  Log->logReward<float>(0.0);
}

void MLGCNSchedStrategy::extractFeatures(SchedBoundary &Zone,
                                         const CandPolicy &ZonePolicy,
					 const RegPressureTracker &RPTracker,
                                         bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  if (DAG->isTrackingPressure()) {
    SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
    VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
  }

  ReadyQueue &Q = Zone.Available;
  int64_t Pos = 0;
  for (SUnit *SU : Q) {
    SchedCandidate Cand(ZonePolicy);
    initCandidate(Cand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    extractCandidateFeatures(Cand, Pos++);
  }
}

void MLGCNSchedStrategy::extractCandidateFeatures(SchedCandidate &TryCand,
						  int64_t Pos) const {
  LLVM_DEBUG(dbgs() << "[MLGCNSchedStrategy::extractFeatures]\n");
  // Set the features at the column 'Idx'.
  // if Candidate is from Bot, Idx = 2 * Pos
  // if Candidate is from Top, Idx = 2 * Pos + 1
  int64_t Idx = 2 * Pos + TryCand.AtTop;
  LLVM_DEBUG(dbgs() << "Index: " << Idx << ", ");
  LLVM_DEBUG(dbgs() << "TryCand.AtTop: " << TryCand.AtTop << "\n");
  LLVM_DEBUG(dbgs() << "RPDelta.Excess: " << TryCand.RPDelta.Excess.getUnitInc() << ", ");
  LLVM_DEBUG(dbgs() << "RPDelta.CurrentMax: " << TryCand.RPDelta.CurrentMax.getUnitInc() << ", ");
  LLVM_DEBUG(dbgs() << "RPDelta.CriticalMax: " << TryCand.RPDelta.CriticalMax.getUnitInc() << "\n");
  LLVM_DEBUG(dbgs() << "SU->Latency: " << TryCand.SU->Latency << ", ");
  LLVM_DEBUG(dbgs() << "SU->getHeight: " << TryCand.SU->getHeight() << ", ");
  LLVM_DEBUG(dbgs() << "SU->getDepth: " << TryCand.SU->getDepth() << "\n");
  LLVM_DEBUG(dbgs() << "SU->NumSuccsLeft: " << TryCand.SU->NumSuccsLeft << ", ");
  LLVM_DEBUG(dbgs() << "SU->NumPredsLeft: " << TryCand.SU->NumPredsLeft << "\n");
  LLVM_DEBUG(dbgs() << "SU->NumSuccs: " << TryCand.SU->NumSuccs << ", ");
  LLVM_DEBUG(dbgs() << "SU->NumPreds: " << TryCand.SU->NumPreds << "\n");

#define SET(ID, TYPE, VAL)						\
  do {                                                                  \
    Runner->getTensor<TYPE>(SchedFeatureIDs::ID)[Idx] = static_cast<TYPE>(VAL); \
  } while (false)
  SET(mask, int64_t, 1);
  SET(is_top, int64_t, TryCand.AtTop);
  SET(is_bot, int64_t, !TryCand.AtTop);
  SET(excess, int64_t, TryCand.RPDelta.Excess.getUnitInc());
  SET(current_max, int64_t, TryCand.RPDelta.CurrentMax.getUnitInc());
  SET(critical_max, int64_t, TryCand.RPDelta.CriticalMax.getUnitInc());
  SET(su_latency, int64_t, TryCand.SU->Latency);
  SET(su_height, int64_t, TryCand.SU->getHeight());
  SET(su_depth, int64_t, TryCand.SU->getDepth());
  SET(su_succs_left, int64_t, TryCand.SU->NumSuccsLeft);
  SET(su_preds_left, int64_t, TryCand.SU->NumPredsLeft);
  SET(su_succs, int64_t, TryCand.SU->NumSuccs);
  SET(su_preds, int64_t, TryCand.SU->NumPreds);
#undef SET

  LLVM_DEBUG(dbgs() << "Finished extracting features!\n");
}


void MLGCNSchedStrategy::extractGlobalFeatures() {
#define SET(ID, TYPE, VAL)						\
  do {									\
    *Runner->getTensor<TYPE>(SchedFeatureIDs::ID) = static_cast<TYPE>(VAL); \
  } while (false)
  SET(sgpr_critical_limit, int64_t, SGPRCriticalLimit);
  SET(vgpr_critical_limit, int64_t, VGPRCriticalLimit);
  SET(sgpr_excess_limit, int64_t, SGPRExcessLimit);
  SET(vgpr_excess_limit, int64_t, VGPRExcessLimit);
#undef SET
}
