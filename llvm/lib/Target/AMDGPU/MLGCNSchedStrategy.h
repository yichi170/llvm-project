//===-- MLGCNSchedStrategy.h - GCN ML Scheduler Strategy -*- C++ -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------------===//
//
/// \file
//
//===---------------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MLGCNSCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_AMDGPU_MLGCNSCHEDSTRATEGY_H

#include "GCNSchedStrategy.h"
#include <cstdint>
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {
class MLGCNSchedStrategy final: public GCNSchedStrategy {
public:
  MLGCNSchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  enum class SchedMode : int { Default, Release, Development };

protected:
  const MLModelRunner &getRunner() const { return *Runner; }

  SUnit *pickNodeByModel();

  void logMLFeatures(int64_t SchedIndex, bool &IsTopNode);

  void extractFeatures(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
		       const RegPressureTracker &RPTracker, bool IsBottomUp);

  void extractCandidateFeatures(SchedCandidate &TryCand, int64_t Pos) const;

  void extractGlobalFeatures();

  // Machine Learning Guided Instruction Scheduling
  std::unique_ptr<MLModelRunner> Runner;
  std::unique_ptr<Logger> Log;
  const SchedMode Mode;
};

} // End namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_MLGCNSCHEDSTRATEGY_H
