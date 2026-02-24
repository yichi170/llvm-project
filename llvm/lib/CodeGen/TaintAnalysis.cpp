//===- TaintAnalysis.cpp - Taint Analysis Pass in Backend ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TaintAnalysis pass which identifies tainted
// registers at the MIR level based on IR function argument attributes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TaintAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "taint-analysis"

AnalysisKey TaintAnalysis::Key;

static cl::opt<std::string>
    TaintOutputFile("taint-output",
                    cl::desc("Output file for tainted instructions (TSV)"),
                    cl::value_desc("file"));

static MemLoc getMemLocFromMMO(const MachineMemOperand &MMO) {
  MemLoc L;
  if (const Value *V =
          MMO.getValue()) { // base address of memory access
                            // :contentReference[oaicite:2]{index=2}
    L.K = MemLoc::IRValue;
    L.V = V;
    return L;
  }
  return L; // Unknown
}

static SmallVector<MemLoc, 2> getMemLocs(const MachineInstr &MI) {
  SmallVector<MemLoc, 2> Locs;
  for (MachineMemOperand *MMO : MI.memoperands()) {
    if (MMO)
      Locs.push_back(getMemLocFromMMO(*MMO));
  }
  return Locs;
}

static bool anyTaintedRegUse(const llvm::MachineInstr &MI,
                             const llvm::TaintState &S) {
  for (const llvm::MachineOperand &MO : MI.uses()) {
    if (MO.isReg()) {
      llvm::Register R = MO.getReg();
      if (R.isValid() && S.isTainted(R))
        return true;
    }
  }
  return false;
}

static void taintAllRegDefs(const llvm::MachineInstr &MI, llvm::TaintState &S,
                            const TargetRegisterInfo *TRI) {
  for (const llvm::MachineOperand &MO : MI.defs()) {
    if (MO.isReg()) {
      llvm::Register R = MO.getReg();
      if (R.isValid()) {
        S.setTainted(R);
        LLVM_DEBUG(dbgs() << "      set " << printReg(MO.getReg(), TRI)
                          << " as tainted\n");
      }
    }
  }
}

static void propagateTaintMI(const llvm::MachineInstr &MI, TaintState &S,
                             const llvm::TargetRegisterInfo *TRI) {
  // -----------------------------------------------------------------------
  // Call instructions: propagate taint to return regs, then clear all
  // implicit clobbers (caller-saved registers overwritten by the callee).
  // -----------------------------------------------------------------------
  if (MI.isCall()) {
    // Conservative intra-procedure assumption: any call can return tainted data.
    // Taint all non-dead defs (explicit defs and live implicit defs = return regs).
    for (const llvm::MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isValid() && !MO.isDead()) {
        S.setTainted(MO.getReg());
        LLVM_DEBUG(dbgs() << "      taint return reg "
                          << printReg(MO.getReg(), TRI) << " (call)\n");
      }
    }
    // Clear dead implicit defs — these are genuine clobbers the callee
    // overwrites and the caller never reads.
    for (const llvm::MachineOperand &MO : MI.implicit_operands()) {
      if (MO.isReg() && MO.isDef() && MO.isDead()) {
        S.clearTainted(MO.getReg());
      }
    }
    return;
  }

  // -----------------------------------------------------------------------
  // Memory propagation state (needed for both store and load paths).
  // -----------------------------------------------------------------------
  SmallVector<MemLoc, 2> Mems = getMemLocs(MI);
  bool HasMem = !Mems.empty();
  bool LoadFromTainted = false;

  // -----------------------------------------------------------------------
  // Reg -> Reg, and Reg -> Mem (store).
  // -----------------------------------------------------------------------
  bool UsesTainted = anyTaintedRegUse(MI, S);

  if (UsesTainted) {
    taintAllRegDefs(MI, S, TRI);
  }

  if (MI.mayStore() && UsesTainted) {
    if (!HasMem) {
      S.UnknownMemTainted = true;
      dbgs() << "      taint unknown mem (store)\n";
    } else {
      for (const MemLoc &L : Mems) {
        S.setTaintedMem(L);
        LLVM_DEBUG({
          if (L.K == MemLoc::IRValue)
            dbgs() << "      taint mem @" << L.V->getName() << " (store)\n";
          else
            dbgs() << "      taint unknown mem (store)\n";
        });
      }
    }
  }

  // -----------------------------------------------------------------------
  // Mem -> Reg (load).
  // -----------------------------------------------------------------------
  if (MI.mayLoad()) {
    if (!HasMem) {
      LoadFromTainted = S.UnknownMemTainted;
    } else {
      for (const MemLoc &L : Mems)
        LoadFromTainted |= S.isTaintedMem(L);
    }

    if (LoadFromTainted) {
      LLVM_DEBUG(dbgs() << "      taint defs (load)\n");
      taintAllRegDefs(MI, S, TRI);
    }
  }

  // -----------------------------------------------------------------------
  // Clean redefinition: if no operand was tainted and this wasn't a load from
  // tainted memory, clear any taint on the non-implicit defs.
  // This handles the case where a physical register is reused and overwritten
  // with a clean value (e.g., $w0 = MOVZWi 42 after $w0 was tainted).
  // -----------------------------------------------------------------------
  if (!UsesTainted && !LoadFromTainted) {
    for (const llvm::MachineOperand &MO : MI.defs()) {
      if (MO.isReg() && !MO.isImplicit() && MO.getReg().isValid()) {
        if (S.isTainted(MO.getReg())) {
          LLVM_DEBUG(dbgs() << "      clear " << printReg(MO.getReg(), TRI)
                            << " (clean redefinition)\n");
        }
        S.clearTainted(MO.getReg());
      }
    }
  }
}

static TaintState propagateTaintMBB(const MachineBasicBlock &MBB,
                                    const TaintState &In,
                                    const TargetRegisterInfo *TRI) {
  TaintState Out = In;
  for (const auto &MI : MBB) {
    propagateTaintMI(MI, Out, TRI);
  }
  return Out;
}

TaintResult TaintAnalysis::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  const Function &F = MF.getFunction();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  LLVM_DEBUG(dbgs() << "\nTaintAnalysis: analyzing function " << F.getName()
                    << "\n");

  // Collect which argument indices are tainted based on IR attributes
  SmallVector<unsigned, 4> TaintedArgIndices;
  for (const Argument &Arg : F.args()) {
    if (Arg.hasAttribute("tainted")) {
      TaintedArgIndices.push_back(Arg.getArgNo());
      LLVM_DEBUG(dbgs() << "  IR arg " << Arg.getArgNo()
                        << " has 'tainted' attribute\n");
    }
  }

  // The analysis needs to work even if there are no tainted arguments.
  // The seed will be empty and no arguments will be marked as tainted;
  // however, call instructions will be marked as tainted if the callee
  // is tainted (currently, conservatively taint all return values).
  //
  // if (TaintedArgIndices.empty()) {
  //   LLVM_DEBUG(dbgs() << "  No tainted arguments found\n");
  //   return TaintResult{TaintState{},
  //                      DenseMap<const MachineBasicBlock *, TaintState>{}};
  // }

  TaintState Seed;

  // Map tainted argument indices to virtual registers via liveins.
  //
  // For simple scalar arguments, the order of liveins corresponds to argument
  // order (based on calling convention). This is a simplification that works
  // for basic integer/pointer arguments.
  unsigned LiveInIdx = 0;
  for (const auto &[PhysReg, VirtReg] : MRI.liveins()) {
    if (llvm::is_contained(TaintedArgIndices, LiveInIdx)) {
      if (VirtReg.isValid()) {
        Seed.setTainted(VirtReg);
        LLVM_DEBUG(dbgs() << "  Marked virtual register "
                          << printReg(VirtReg, TRI) << " as tainted (from arg "
                          << LiveInIdx << ", phys " << printReg(PhysReg, TRI)
                          << ")\n");
      } else {
        Seed.setTainted(PhysReg);
        LLVM_DEBUG(dbgs() << "  Marked physical register "
                          << printReg(PhysReg, TRI) << " as tainted (from arg "
                          << LiveInIdx << ")\n");
      }
    }
    ++LiveInIdx;
  }

  DenseMap<const MachineBasicBlock *, TaintState> IN, OUT;

  for (auto &MBB : MF) {
    IN[&MBB] = TaintState{};
    OUT[&MBB] = TaintState{};
  }

  IN[&MF.front()] = Seed;
  SmallVector<const llvm::MachineBasicBlock *, 32> WorkQ;
  SmallPtrSet<const llvm::MachineBasicBlock *, 32> InQ;

  auto push = [&](const llvm::MachineBasicBlock *B) {
    if (InQ.insert(B).second)
      WorkQ.push_back(B);
  };

  push(&MF.front());

  while (!WorkQ.empty()) {
    const llvm::MachineBasicBlock *B = WorkQ.pop_back_val();
    InQ.erase(B);

    LLVM_DEBUG(dbgs() << "    " << printMBBReference(*B) << "\n");

    // Recompute IN[B] from preds, but preserve seed for entry.
    TaintState NewIn;
    if (B == &MF.front()) {
      NewIn = Seed;
    } else {
      bool First = true;
      for (const llvm::MachineBasicBlock *P : B->predecessors()) {
        if (First) {
          NewIn = OUT[P];
          First = false;
        } else {
          NewIn.join(OUT[P]);
        }
      }
      if (First) {
        // Unreachable block: keep empty IN
        NewIn = TaintState{};
      }
    }

    // If IN changed, update and recompute OUT
    bool InChanged = (NewIn != IN[B]);
    if (InChanged)
      IN[B] = std::move(NewIn);

    // Compute OUT from IN
    TaintState NewOut = propagateTaintMBB(*B, IN[B], TRI);

    if (NewOut != OUT[B]) {
      OUT[B] = std::move(NewOut);
      for (const llvm::MachineBasicBlock *S : B->successors())
        push(S);
    } else if (InChanged) {
      // IN changed but OUT did not (rare but possible); still safe to push
      // succs
      for (const llvm::MachineBasicBlock *S : B->successors())
        push(S);
    }
  }

  TaintState Result;

  for (auto &MBB : MF) {
    Result.join(OUT[&MBB]);
  }

  LLVM_DEBUG(dbgs() << "Total tainted regs: " << Result.countRegs()
                    << ", tainted FIs: " << Result.countFIs()
                    << ", total: " << Result.count() << "\n");

  return TaintResult{std::move(Result), std::move(IN)};
}

/// Shared implementation: run taint analysis on MF and export results.
static void runTaintAnalysisAndExport(MachineFunction &MF) {
  // Run the core analysis.
  MachineFunctionAnalysisManager DummyMFAM;
  TaintAnalysis TA;
  TaintResult TR = TA.run(MF, DummyMFAM);

  LLVM_DEBUG({
    if (!TR.Merged.empty()) {
      dbgs() << "TaintAnalysisPass: " << MF.getName() << " has "
             << TR.Merged.count() << " tainted register(s)\n";
    }
  });

  // Export tainted instructions to file if requested.
  if (TaintOutputFile.empty() || TR.Merged.empty())
    return;

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // Open file in append mode so multiple functions accumulate.
  std::error_code EC;
  raw_fd_ostream OS(TaintOutputFile, EC, sys::fs::OF_Append);
  if (EC) {
    errs() << "Error opening taint output file: " << EC.message() << "\n";
    return;
  }

  // Cache source file contents: filename -> vector of lines.
  // MemoryBuffers must stay alive so StringRefs into them remain valid.
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> FileBufs;
  StringMap<SmallVector<StringRef, 0>> FileCache;
  auto getSourceLine = [&](StringRef Filename, unsigned Line) -> StringRef {
    auto It = FileCache.find(Filename);
    if (It == FileCache.end()) {
      auto BufOrErr = MemoryBuffer::getFile(Filename);
      if (!BufOrErr) {
        FileCache[Filename] = {};
        return "";
      }
      StringRef Contents = BufOrErr.get()->getBuffer();
      FileBufs.push_back(std::move(BufOrErr.get()));
      SmallVector<StringRef, 0> Lines;
      Contents.split(Lines, '\n');
      FileCache[Filename] = std::move(Lines);
      It = FileCache.find(Filename);
    }
    const auto &Lines = It->second;
    if (Line == 0 || Line > Lines.size())
      return "";
    return Lines[Line - 1]; // 1-indexed to 0-indexed.
  };

  OS << "# Function: " << MF.getName() << "\n";

  // Deduplicate by (filename, line) to avoid repeating the same source line.
  DenseSet<std::pair<unsigned, unsigned>> SeenLines; // hash of filename + line
  // Storage for full path strings so StringRefs into them stay valid.
  SmallVector<std::string, 8> PathStorage;

  for (const auto &MBB : MF) {
    // Start from the entry state for this BB.
    auto It = TR.IN.find(&MBB);
    TaintState S = (It != TR.IN.end()) ? It->second : TaintState{};

    for (const auto &MI : MBB) {
      bool UsesTainted = anyTaintedRegUse(MI, S);

      // Propagate taint through this instruction.
      propagateTaintMI(MI, S, TRI);

      // Check if any def became tainted.
      bool DefsTainted = false;
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isReg() && MO.getReg().isValid() && S.isTainted(MO.getReg())) {
          DefsTainted = true;
          break;
        }
      }

      if (!UsesTainted && !DefsTainted)
        continue;

      // Record this instruction in TaintedInstrs for future instrumentation.
      TR.TaintedInstrs.insert(&MI);

      // Only export if debug info is available.
      const DebugLoc &DL = MI.getDebugLoc();
      if (!DL)
        continue;

      StringRef Filename = "";
      if (auto *Scope = dyn_cast<DIScope>(DL->getScope())) {
        // Build full path: directory + filename.
        if (auto *File = Scope->getFile()) {
          SmallString<256> FullPath(File->getDirectory());
          sys::path::append(FullPath, File->getFilename());
          PathStorage.emplace_back(FullPath.str());
          Filename = PathStorage.back();
        }
      }
      unsigned Line = DL.getLine();

      if (Filename.empty() || Line == 0)
        continue;

      // Deduplicate by filename hash + line number.
      auto Key = std::make_pair((unsigned)llvm::hash_value(Filename), Line);
      if (!SeenLines.insert(Key).second)
        continue;

      StringRef SrcLine = getSourceLine(Filename, Line);
      OS << Line << ": " << SrcLine.ltrim() << "\n";
    }
  }
}

// ===----------------------------------------------------------------------===//
// New PM pass
// ===----------------------------------------------------------------------===//

PreservedAnalyses TaintAnalysisPass::run(MachineFunction &MF,
                                         MachineFunctionAnalysisManager &MFAM) {
  runTaintAnalysisAndExport(MF);
  return getMachineFunctionPassPreservedAnalyses();
}

// ===----------------------------------------------------------------------===//
// Legacy PM pass
// ===----------------------------------------------------------------------===//

namespace {
struct TaintAnalysisLegacy : public MachineFunctionPass {
  static char ID;
  TaintAnalysisLegacy() : MachineFunctionPass(ID) {
    initializeTaintAnalysisLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    runTaintAnalysisAndExport(MF);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "Taint Analysis"; }
};
} // end anonymous namespace

char TaintAnalysisLegacy::ID = 0;
char &llvm::TaintAnalysisLegacyID = TaintAnalysisLegacy::ID;

INITIALIZE_PASS(TaintAnalysisLegacy, DEBUG_TYPE, "Taint Analysis Pass", false,
                false)

MachineFunctionPass *llvm::createTaintAnalysisLegacyPass() {
  return new TaintAnalysisLegacy();
}
