//===- llvm/CodeGen/TaintAnalysis.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TaintAnalysis pass which identifies tainted registers
// in MIR based on "tainted" attributes set on IR function arguments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TAINTANALYSIS_H
#define LLVM_CODEGEN_TAINTANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

struct MemLoc {
  enum Kind { IRValue, Unknown } K = Unknown;
  const Value *V = nullptr; // base pointer value
};

/// TaintState holds the result of taint analysis for a MachineFunction.
/// It tracks which virtual registers are considered tainted.
struct TaintState {
  SparseBitVector<> TaintedRegs;
  DenseSet<int> TaintedFrameIdx;
  DenseSet<const Value *> TaintedMemVals;
  bool UnknownMemTainted = false;

public:
  bool operator==(const TaintState &O) const {
    return TaintedRegs == O.TaintedRegs &&
           TaintedFrameIdx == O.TaintedFrameIdx &&
           TaintedMemVals == O.TaintedMemVals &&
           UnknownMemTainted == O.UnknownMemTainted;
  }

  bool operator!=(const TaintState &O) const { return !(*this == O); }

  void join(const TaintState &O) {
    TaintedRegs |= O.TaintedRegs;
    for (const auto &FI : O.TaintedFrameIdx) {
      TaintedFrameIdx.insert(FI);
    }
    for (const Value *V : O.TaintedMemVals) {
      TaintedMemVals.insert(V);
    }
    UnknownMemTainted |= O.UnknownMemTainted;
  }

  /// Check if a register is tainted.
  bool isTainted(Register R) const {
    return R.isValid() && TaintedRegs.test(R.id());
  }

  /// Mark a register as tainted.
  void setTainted(Register R) {
    if (R.isValid())
      TaintedRegs.set(R.id());
  }

  /// Remove a register from the tainted set (clean redefinition or clobber).
  void clearTainted(Register R) {
    if (R.isValid())
      TaintedRegs.reset(R.id());
  }

  bool isTaintedFI(int FI) const { return TaintedFrameIdx.contains(FI); }
  void setTaintedFI(int FI) { TaintedFrameIdx.insert(FI); }
  void clearTaintedFI(int FI) { TaintedFrameIdx.erase(FI); }

  bool isTaintedMem(const MemLoc &L) const {
    if (L.K == MemLoc::IRValue)
      return TaintedMemVals.contains(L.V);
    return UnknownMemTainted;
  }

  void setTaintedMem(const MemLoc &L) {
    if (L.K == MemLoc::IRValue)
      TaintedMemVals.insert(L.V);
    else
      UnknownMemTainted = true;
  }

  bool emptyRegs() const { return TaintedRegs.empty(); }
  bool emptyFIs() const { return TaintedFrameIdx.empty(); }
  bool empty() const { return emptyRegs() && emptyFIs(); }

  unsigned countRegs() const { return TaintedRegs.count(); }
  unsigned countFIs() const { return (unsigned)TaintedFrameIdx.size(); }
  unsigned count() const { return countRegs() + countFIs(); }
};

/// TaintResult holds both the merged taint state and per-BB entry states.
/// The per-BB IN map is needed by the export pass to replay taint propagation
/// instruction-by-instruction and determine which specific instructions are
/// tainted.
struct TaintResult {
  TaintState Merged;
  DenseMap<const MachineBasicBlock *, TaintState> IN;
  /// Instructions that touch tainted data (use or define a tainted value).
  /// Populated during the export/replay walk; not part of the lattice.
  /// Intended for use by future instrumentation passes.
  DenseSet<const MachineInstr *> TaintedInstrs;
};

/// TaintAnalysis is a MachineFunction analysis that computes which registers
/// are tainted based on IR function argument attributes.
class TaintAnalysis : public AnalysisInfoMixin<TaintAnalysis> {
  friend AnalysisInfoMixin<TaintAnalysis>;
  static AnalysisKey Key;

public:
  using Result = TaintResult;
  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

/// TaintAnalysisPass is a MachineFunction pass that runs TaintAnalysis
/// and can be used to trigger the analysis in the pass pipeline.
class TaintAnalysisPass : public PassInfoMixin<TaintAnalysisPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_TAINTANALYSIS_H
