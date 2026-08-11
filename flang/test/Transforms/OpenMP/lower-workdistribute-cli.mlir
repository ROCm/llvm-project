// Verifies that the `implicit` pass option on `lower-workdistribute` is
// recognized via both fir-opt invocation styles. The pass currently performs
// no IR mutation when there is no `omp.workdistribute` in the module, so the
// test only checks that the module survives every option state.

// RUN: fir-opt --pass-pipeline='builtin.module(lower-workdistribute{implicit=host})' %s | FileCheck %s
// RUN: fir-opt --lower-workdistribute='implicit=host' %s | FileCheck %s
// RUN: fir-opt --lower-workdistribute='implicit=device' %s | FileCheck %s
// RUN: fir-opt --lower-workdistribute='implicit=none' %s | FileCheck %s
// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL: func.func @noop
// CHECK-NEXT:    return
module {
  func.func @noop() {
    return
  }
}
