// On AMDGPU, a member whose type has no non-static data members keeps its
// storage in the IR record.  The de facto device convention for passing and
// returning aggregates is derived from the IR record, and ROCgdb reconstructs
// it from the DWARF type, so turning such a member into explicit padding would
// shift the registers assigned to the members that follow it.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=AMDGCN

struct OnlyStatic {
  static int something;
};

struct WithStaticFields {
  int a[2];
  OnlyStatic sub;
  float b;
  double d;
};

// HOST: %struct.WithStaticFields = type { [2 x i32], [4 x i8], float, double }
// AMDGCN: %struct.WithStaticFields = type { [2 x i32], %struct.OnlyStatic, float, double }

WithStaticFields returnWithStatic() {
  WithStaticFields r{};
  r.b = 3.14f;
  r.d = 1.60218e-19;
  return r;
}

// A member marked [[no_unique_address]] is dropped on both targets.
struct WithNoUniqueAddress {
  int a[2];
  [[no_unique_address]] OnlyStatic sub;
  float b;
  double d;
};

// HOST: %struct.WithNoUniqueAddress = type { [2 x i32], float, double }
// AMDGCN: %struct.WithNoUniqueAddress = type { [2 x i32], float, double }

WithNoUniqueAddress returnNoUniqueAddress() {
  WithNoUniqueAddress r{};
  return r;
}
