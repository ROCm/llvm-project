// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --version 5
// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

#define __device__ __attribute__((device))

// CHECK-LABEL: define dso_local void @_Z7kernel1v(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] !dbg [[DBG7:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[UNION_ANON:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META12:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[UNION_ANON]])), [[META21:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META14:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[UNION_ANON]])), [[META21]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META16:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[UNION_ANON]])), [[META21]])
// CHECK-NEXT:    ret void, !dbg [[DBG22:![0-9]+]]
//
__device__ void kernel1() {
  union { int x; float f; };
}

//.
// CHECK: [[META0:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: [[META1:![0-9]+]], producer: "{{.*}}clang version {{.*}}", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
// CHECK: [[META1]] = !DIFile(filename: "{{.*}}<stdin>", directory: {{.*}})
// CHECK: [[DBG7]] = distinct !DISubprogram(name: "kernel1", linkageName: "_Z7kernel1v", scope: [[META8:![0-9]+]], file: [[META8]], line: 17, type: [[META9:![0-9]+]], scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META11:![0-9]+]])
// CHECK: [[META8]] = !DIFile(filename: "{{.*}}debug-info-anonymous-union-heterogeneous-dwarf.hip", directory: {{.*}})
// CHECK: [[META9]] = !DISubroutineType(types: [[META10:![0-9]+]])
// CHECK: [[META10]] = !{null}
// CHECK: [[META11]] = !{[[META12]], [[META14]], [[META16]]}
// CHECK: [[META12]] = !DILocalVariable(name: "x", scope: [[DBG7]], type: [[META13:![0-9]+]], flags: DIFlagArtificial)
// CHECK: [[META13]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: [[META14]] = !DILocalVariable(name: "f", scope: [[DBG7]], type: [[META15:![0-9]+]], flags: DIFlagArtificial)
// CHECK: [[META15]] = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
// CHECK: [[META16]] = !DILocalVariable(scope: [[DBG7]], type: [[META17:![0-9]+]], flags: DIFlagArtificial)
// CHECK: [[META17]] = distinct !DICompositeType(tag: DW_TAG_union_type, scope: [[DBG7]], file: [[META8]], line: 18, size: 32, flags: DIFlagExportSymbols | DIFlagTypePassByValue, elements: [[META18:![0-9]+]])
// CHECK: [[META18]] = !{[[META19:![0-9]+]], [[META20:![0-9]+]]}
// CHECK: [[META19]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META17]], file: [[META8]], line: 18, baseType: [[META13]], size: 32)
// CHECK: [[META20]] = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: [[META17]], file: [[META8]], line: 18, baseType: [[META15]], size: 32)
// CHECK: [[META21]] = !DILocation(line: 0, scope: [[DBG7]])
// CHECK: [[DBG22]] = !DILocation(line: 19, column: 1, scope: [[DBG7]])
//.
