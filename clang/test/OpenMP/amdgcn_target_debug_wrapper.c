// Scalars captured by copy are widened to uintptr_t in the outlined function
// signature, and that widened type reaches the debug info as well. When debug
// info is requested, the user code is therefore emitted into a companion
// `..._debug__` function that carries the original parameter types, while the
// kernel entry point keeps the widened ABI signature. AMDGCN was excluded from
// this for years, which described a firstprivate `int` as `unsigned long` in
// DWARF, so check that the device now gets the companion function like every
// other target.

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -debug-info-kind=limited \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// The companion function is only emitted when variable and type info is
// requested, which means -debug-info-kind=constructor and above. Below that
// there is no description of the parameters for the widened signature to get
// wrong, so the kernel itself holds the user code and no companion is emitted.
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -debug-info-kind=line-tables-only \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NO-WRAPPER
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=NO-WRAPPER

// expected-no-diagnostics

int test(void) {
  int arr[8];
  int fp = 42;

#pragma omp target map(tofrom: arr[0 : 8]) firstprivate(fp)
  { arr[0] = fp; }

  return arr[0];
}

// The user code, taking the scalar with its original width.
// CHECK: define internal void @[[DEBUG_FN:__omp_offloading_[0-9a-z_]+_test_l[0-9]+_debug__]](ptr addrspace(1) noalias noundef %{{[a-z0-9_.]+}}, i32 noundef %{{[a-z0-9_.]+}}, ptr noalias noundef %{{[a-z0-9_.]+}}) #{{[0-9]+}} !dbg [[DEBUG_SP:![0-9]+]]
// CHECK: #dbg_declare(ptr %{{[a-z0-9_.]+}}, [[ARR_VAR:![0-9]+]],

// The entry point, keeping the widened signature and calling the above.
// CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-z_]+}}_test_l{{[0-9]+}}(ptr noundef nonnull align 4 dereferenceable(32) %{{[a-z0-9_.]+}}, i64 noundef %{{[a-z0-9_.]+}}, ptr noalias noundef %{{[a-z0-9_.]+}}) #{{[0-9]+}} !dbg [[KERNEL_SP:![0-9]+]]
// CHECK: call void @[[DEBUG_FN]](ptr addrspace(1) %{{[0-9]+}}, i32 %{{[0-9]+}}, ptr %{{[0-9]+}})

// `fp` is described as `int` in the function holding the user code, which is
// the frame a debugger stops in. The scope and type operands pin each variable
// to the function it belongs to, so the order these nodes are emitted in does
// not matter.
// CHECK-DAG: [[DEBUG_SP]] = distinct !DISubprogram(name: "[[DEBUG_FN]]"
// CHECK-DAG: [[INT_TY:![0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK-DAG: !DILocalVariable(name: "fp", arg: 2, scope: [[DEBUG_SP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: [[INT_TY]])
// CHECK-DAG: [[ARR_VAR]] = !DILocalVariable(name: "arr", arg: 1, scope: [[DEBUG_SP]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: [[ARR_REF_TY:![0-9]+]])
// CHECK-DAG: [[ARR_REF_TY]] = !DIDerivedType(tag: DW_TAG_reference_type, baseType: [[ARR_TY:![0-9]+]]
// CHECK-DAG: [[ARR_TY]] = !DICompositeType(tag: DW_TAG_array_type, baseType: [[INT_TY]], size: 256, elements: [[ARR_ELEMS:![0-9]+]]
// CHECK-DAG: [[ARR_ELEMS]] = !{[[ARR_SUBRANGE:![0-9]+]]}
// CHECK-DAG: [[ARR_SUBRANGE]] = !DISubrange(count: 8)

// The entry point still describes it as the widened type, marked artificial,
// which is accurate for that frame.
// CHECK-DAG: [[KERNEL_SP]] = distinct !DISubprogram(name: "__omp_offloading_{{[0-9a-z_]+}}_test_l{{[0-9]+}}"
// CHECK-DAG: [[LONG_TY:![0-9]+]] = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
// CHECK-DAG: !DILocalVariable(name: "fp", arg: 2, scope: [[KERNEL_SP]], type: [[LONG_TY]], flags: DIFlagArtificial)

// The kernel holds the user code, and nothing anywhere in the module refers to
// a companion function.
// NO-WRAPPER-NOT: _debug__
// NO-WRAPPER: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[0-9a-z_]+}}_test_l{{[0-9]+}}(ptr noundef nonnull align 4 dereferenceable(32) %{{[a-z0-9_.]+}}, i64 noundef %{{[a-z0-9_.]+}}, ptr noalias noundef %{{[a-z0-9_.]+}}) #{{[0-9]+}}
// NO-WRAPPER-NOT: _debug__
