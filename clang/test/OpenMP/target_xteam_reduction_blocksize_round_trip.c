// Explicitly requesting a cross-team reduction block size must not make the
// generated -cc1 arguments disagree with the parsed ones. '-round-trip-args'
// forces the check in builds without assertions as well.

// RUN: %clang_cc1 -round-trip-args -fopenmp -fopenmp-target-xteam-reduction-blocksize=1024 \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s -allow-empty
// RUN: %clang_cc1 -round-trip-args -fopenmp -fopenmp-target-xteam-reduction-blocksize=512 \
// RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s -allow-empty
// RUN: %clang_cc1 -round-trip-args -fopenmp -emit-llvm -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s -allow-empty

// CHECK-NOT: error:

void f(void) {}
