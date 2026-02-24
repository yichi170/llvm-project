// RUN: clang -g -S -emit-llvm %s -o %t.ll
// RUN: opt -passes=taint-annotate -taint-src=%S/Inputs/secret_call_flow.txt %t.ll -S -o %t.annotated.ll
// RUN: llc -O0 -taint-output=%t.taint.txt %t.annotated.ll -o /dev/null
// RUN: FileCheck %s --input-file=%t.taint.txt

// Test: taint flows from tainted_param's argument through call sites in main.
// Conservative intra-procedure analysis: all call return values are tainted.
// tainted_param has its first argument annotated as tainted via secret_call_flow.txt.

// CHECK: # Function: tainted_param
// CHECK-NEXT: [[@LINE+2]]: return a + b;
int tainted_param(int a, int b) {
    return a + b;
}

int clean_param(int a, int b) {
    return a + b;
}

int clean_param_with_tainted_arg(int a, int b) {
    return a + b;
}

// CHECK: # Function: main
// CHECK-NEXT: [[@LINE+8]]: int tainted = tainted_param(a, b);
// CHECK-NEXT: [[@LINE+8]]: int clean = clean_param(c, d);
// CHECK-NEXT: [[@LINE+8]]: int tainted2 = clean_param_with_tainted_arg(tainted, c);
int main() {
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    int tainted = tainted_param(a, b);
    int clean = clean_param(c, d);
    int tainted2 = clean_param_with_tainted_arg(tainted, c);
    return tainted + clean + tainted2;
}
