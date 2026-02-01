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

#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

/// TaintInfo holds the result of taint analysis for a MachineFunction.
/// It tracks which virtual registers are considered tainted.
class TaintInfo {
  SparseBitVector<> TaintedRegs;

public:
  /// Check if a register is tainted.
  bool isTainted(Register R) const { return TaintedRegs.test(R.id()); }

  /// Mark a register as tainted.
  void setTainted(Register R) { TaintedRegs.set(R.id()); }

  /// Check if no registers are tainted.
  bool empty() const { return TaintedRegs.empty(); }

  /// Get the number of tainted registers.
  unsigned count() const { return TaintedRegs.count(); }

  /// Iterator access to tainted register IDs.
  using iterator = SparseBitVector<>::iterator;
  iterator begin() const { return TaintedRegs.begin(); }
  iterator end() const { return TaintedRegs.end(); }
};

/// TaintAnalysis is a MachineFunction analysis that computes which registers
/// are tainted based on IR function argument attributes.
class TaintAnalysis : public AnalysisInfoMixin<TaintAnalysis> {
  friend AnalysisInfoMixin<TaintAnalysis>;
  static AnalysisKey Key;

public:
  using Result = TaintInfo;
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
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_TAINTANALYSIS_H
