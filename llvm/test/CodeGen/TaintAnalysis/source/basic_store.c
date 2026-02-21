// RUN: clang -O0 -g -S -emit-llvm %s -o %t.ll
// RUN: opt -passes=taint-annotate -taint-src=%S/Inputs/secret.txt %t.ll -S -o %t.annotated.ll
// RUN: llc -O0 -taint-output=%t.taint.txt %t.annotated.ll -filetype=null
// RUN: FileCheck %s < %t.taint.txt

// CHECK: # Function: simple_store
// CHECK-NEXT: [[@LINE+4]]: int x = secret + 1;
// CHECK-NEXT: [[@LINE+4]]: arr[x] = 10;

void simple_store(int secret, int *arr) {
    int x = secret + 1;
    arr[x] = 10;
}
