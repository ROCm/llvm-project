// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --version 5
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x c++ -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

struct S0 {
  unsigned int x : 16;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS0v(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] !dbg [[DBG6:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S0:%.*]] = alloca [[STRUCT_S0:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S0]], align 4, addrspace(5)
// CHECK-NEXT:    [[S0_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S0]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S0]], [[META11:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S0]])), [[META17:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META18:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S0]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META19:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META20:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S0]]), DIOpConstant(i32 16), DIOpBitOffset(i32)), [[META21:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S0_ASCAST]], i64 4, i1 false), !dbg [[DBG22:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG23:![0-9]+]]
//
void fS0() {
  S0 s0;
  auto [a, b] = s0;
}

struct S1 {
  unsigned int x : 8;
  unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS1v(
// CHECK-SAME: ) #[[ATTR0]] !dbg [[DBG24:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S1:%.*]] = alloca [[STRUCT_S1:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S1]], align 4, addrspace(5)
// CHECK-NEXT:    [[S1_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S1]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S1]], [[META26:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S1]])), [[META31:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META32:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S1]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META33:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META34:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S1]]), DIOpConstant(i32 8), DIOpBitOffset(i32)), [[META35:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S1_ASCAST]], i64 4, i1 false), !dbg [[DBG36:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG37:![0-9]+]]
//
void fS1() {
  S1 s1;
  auto [a, b] = s1;
}

struct S2 {
  unsigned int x : 8;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS2v(
// CHECK-SAME: ) #[[ATTR0]] !dbg [[DBG38:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S2:%.*]] = alloca [[STRUCT_S2:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S2]], align 4, addrspace(5)
// CHECK-NEXT:    [[S2_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S2]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S2]], [[META40:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S2]])), [[META45:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META46:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S2]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META47:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META48:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S2]]), DIOpConstant(i32 8), DIOpBitOffset(i32)), [[META49:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S2_ASCAST]], i64 4, i1 false), !dbg [[DBG50:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG51:![0-9]+]]
//
void fS2() {
  S2 s2;
  auto [a, b] = s2;
}

struct S3 {
  unsigned int x : 16;
  unsigned int y : 32;
};

// CHECK-LABEL: define dso_local void @_Z3fS3v(
// CHECK-SAME: ) #[[ATTR0]] !dbg [[DBG52:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S3:%.*]] = alloca [[STRUCT_S3:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S3]], align 4, addrspace(5)
// CHECK-NEXT:    [[S3_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S3]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S3]], [[META54:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S3]])), [[META59:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META60:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S3]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META61:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META62:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S3]]), DIOpConstant(i32 32), DIOpBitOffset(i32)), [[META63:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S3_ASCAST]], i64 8, i1 false), !dbg [[DBG64:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG65:![0-9]+]]
//
void fS3() {
  S3 s3;
  auto [a, b] = s3;
}

struct S4 {
  unsigned int x : 16;
  unsigned : 0;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS4v(
// CHECK-SAME: ) #[[ATTR0]] !dbg [[DBG66:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S4:%.*]] = alloca [[STRUCT_S4:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S4]], align 4, addrspace(5)
// CHECK-NEXT:    [[S4_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S4]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S4]], [[META68:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S4]])), [[META74:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META75:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S4]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META76:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META77:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S4]]), DIOpConstant(i32 32), DIOpBitOffset(i32)), [[META78:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S4_ASCAST]], i64 8, i1 false), !dbg [[DBG79:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG80:![0-9]+]]
//
void fS4() {
  S4 s4;
  auto [a, b] = s4;
}

// It's currently not possible to produce complete debug information for the following cases.
// Confirm that no wrong debug info is output.
// Once this is implemented, these tests should be amended.
struct S5 {
  unsigned int x : 15;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS5v(
// CHECK-SAME: ) #[[ATTR0]] !dbg [[DBG81:![0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[S5:%.*]] = alloca [[STRUCT_S5:%.*]], align 4, addrspace(5)
// CHECK-NEXT:    [[TMP0:%.*]] = alloca [[STRUCT_S5]], align 4, addrspace(5)
// CHECK-NEXT:    [[S5_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[S5]] to ptr
// CHECK-NEXT:    [[TMP1:%.*]] = addrspacecast ptr addrspace(5) [[TMP0]] to ptr
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[S5]], [[META83:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S5]])), [[META88:![0-9]+]])
// CHECK-NEXT:      #dbg_declare(ptr addrspace(5) [[TMP0]], [[META89:![0-9]+]], !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref([[STRUCT_S5]]), DIOpConstant(i32 0), DIOpBitOffset(i32)), [[META90:![0-9]+]])
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP1]], ptr align 4 [[S5_ASCAST]], i64 4, i1 false), !dbg [[DBG91:![0-9]+]]
// CHECK-NEXT:    ret void, !dbg [[DBG92:![0-9]+]]
//
void fS5() {
  S5 s5;
  auto [a, b] = s5;
}

// Currently, LLVM when it emits the structured binding for a bitfield it also emits the DIExpression as an i32 (which mismaches the bitfield width)






//.
// CHECK: [[META0:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: [[META1:![0-9]+]], producer: "{{.*}}clang version {{.*}}", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
// CHECK: [[META1]] = !DIFile(filename: "{{.*}}<stdin>", directory: {{.*}})
// CHECK: [[DBG6]] = distinct !DISubprogram(name: "fS0", linkageName: "_Z3fS0v", scope: [[META7:![0-9]+]], file: [[META7]], line: 22, type: [[META8:![0-9]+]], scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META10:![0-9]+]])
// CHECK: [[META7]] = !DIFile(filename: "{{.*}}heterogeneous-debug-info-structured-binding-bitfield.cpp", directory: {{.*}})
// CHECK: [[META8]] = !DISubroutineType(types: [[META9:![0-9]+]])
// CHECK: [[META9]] = !{null}
// CHECK: [[META10]] = !{[[META11]]}
// CHECK: [[META11]] = !DILocalVariable(name: "s0", scope: [[DBG6]], file: [[META7]], line: 23, type: [[META12:![0-9]+]])
// CHECK: [[META12]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: [[META7]], line: 4, size: 32, flags: DIFlagTypePassByValue, elements: [[META13:![0-9]+]], identifier: "_ZTS2S0")
// CHECK: [[META13]] = !{[[META14:![0-9]+]], [[META16:![0-9]+]]}
// CHECK: [[META14]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META12]], file: [[META7]], line: 5, baseType: [[META15:![0-9]+]], size: 16, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META15]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
// CHECK: [[META16]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META12]], file: [[META7]], line: 6, baseType: [[META15]], size: 16, offset: 16, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META17]] = !DILocation(line: 23, column: 6, scope: [[DBG6]])
// CHECK: [[META18]] = !DILocalVariable(name: "a", scope: [[DBG6]], file: [[META7]], line: 24, type: [[META15]])
// CHECK: [[META19]] = !DILocation(line: 24, column: 9, scope: [[DBG6]])
// CHECK: [[META20]] = !DILocalVariable(name: "b", scope: [[DBG6]], file: [[META7]], line: 24, type: [[META15]])
// CHECK: [[META21]] = !DILocation(line: 24, column: 12, scope: [[DBG6]])
// CHECK: [[DBG22]] = !DILocation(line: 24, column: 17, scope: [[DBG6]])
// CHECK: [[DBG23]] = !DILocation(line: 25, column: 1, scope: [[DBG6]])
// CHECK: [[DBG24]] = distinct !DISubprogram(name: "fS1", linkageName: "_Z3fS1v", scope: [[META7]], file: [[META7]], line: 45, type: [[META8]], scopeLine: 45, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META25:![0-9]+]])
// CHECK: [[META25]] = !{[[META26]]}
// CHECK: [[META26]] = !DILocalVariable(name: "s1", scope: [[DBG24]], file: [[META7]], line: 46, type: [[META27:![0-9]+]])
// CHECK: [[META27]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S1", file: [[META7]], line: 27, size: 32, flags: DIFlagTypePassByValue, elements: [[META28:![0-9]+]], identifier: "_ZTS2S1")
// CHECK: [[META28]] = !{[[META29:![0-9]+]], [[META30:![0-9]+]]}
// CHECK: [[META29]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META27]], file: [[META7]], line: 28, baseType: [[META15]], size: 8, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META30]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META27]], file: [[META7]], line: 29, baseType: [[META15]], size: 8, offset: 8, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META31]] = !DILocation(line: 46, column: 6, scope: [[DBG24]])
// CHECK: [[META32]] = !DILocalVariable(name: "a", scope: [[DBG24]], file: [[META7]], line: 47, type: [[META15]])
// CHECK: [[META33]] = !DILocation(line: 47, column: 9, scope: [[DBG24]])
// CHECK: [[META34]] = !DILocalVariable(name: "b", scope: [[DBG24]], file: [[META7]], line: 47, type: [[META15]])
// CHECK: [[META35]] = !DILocation(line: 47, column: 12, scope: [[DBG24]])
// CHECK: [[DBG36]] = !DILocation(line: 47, column: 17, scope: [[DBG24]])
// CHECK: [[DBG37]] = !DILocation(line: 48, column: 1, scope: [[DBG24]])
// CHECK: [[DBG38]] = distinct !DISubprogram(name: "fS2", linkageName: "_Z3fS2v", scope: [[META7]], file: [[META7]], line: 68, type: [[META8]], scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META39:![0-9]+]])
// CHECK: [[META39]] = !{[[META40]]}
// CHECK: [[META40]] = !DILocalVariable(name: "s2", scope: [[DBG38]], file: [[META7]], line: 69, type: [[META41:![0-9]+]])
// CHECK: [[META41]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S2", file: [[META7]], line: 50, size: 32, flags: DIFlagTypePassByValue, elements: [[META42:![0-9]+]], identifier: "_ZTS2S2")
// CHECK: [[META42]] = !{[[META43:![0-9]+]], [[META44:![0-9]+]]}
// CHECK: [[META43]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META41]], file: [[META7]], line: 51, baseType: [[META15]], size: 8, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META44]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META41]], file: [[META7]], line: 52, baseType: [[META15]], size: 16, offset: 8, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META45]] = !DILocation(line: 69, column: 6, scope: [[DBG38]])
// CHECK: [[META46]] = !DILocalVariable(name: "a", scope: [[DBG38]], file: [[META7]], line: 70, type: [[META15]])
// CHECK: [[META47]] = !DILocation(line: 70, column: 9, scope: [[DBG38]])
// CHECK: [[META48]] = !DILocalVariable(name: "b", scope: [[DBG38]], file: [[META7]], line: 70, type: [[META15]])
// CHECK: [[META49]] = !DILocation(line: 70, column: 12, scope: [[DBG38]])
// CHECK: [[DBG50]] = !DILocation(line: 70, column: 17, scope: [[DBG38]])
// CHECK: [[DBG51]] = !DILocation(line: 71, column: 1, scope: [[DBG38]])
// CHECK: [[DBG52]] = distinct !DISubprogram(name: "fS3", linkageName: "_Z3fS3v", scope: [[META7]], file: [[META7]], line: 91, type: [[META8]], scopeLine: 91, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META53:![0-9]+]])
// CHECK: [[META53]] = !{[[META54]]}
// CHECK: [[META54]] = !DILocalVariable(name: "s3", scope: [[DBG52]], file: [[META7]], line: 92, type: [[META55:![0-9]+]])
// CHECK: [[META55]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S3", file: [[META7]], line: 73, size: 64, flags: DIFlagTypePassByValue, elements: [[META56:![0-9]+]], identifier: "_ZTS2S3")
// CHECK: [[META56]] = !{[[META57:![0-9]+]], [[META58:![0-9]+]]}
// CHECK: [[META57]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META55]], file: [[META7]], line: 74, baseType: [[META15]], size: 16, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META58]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META55]], file: [[META7]], line: 75, baseType: [[META15]], size: 32, offset: 32, flags: DIFlagBitField, extraData: i64 32)
// CHECK: [[META59]] = !DILocation(line: 92, column: 6, scope: [[DBG52]])
// CHECK: [[META60]] = !DILocalVariable(name: "a", scope: [[DBG52]], file: [[META7]], line: 93, type: [[META15]])
// CHECK: [[META61]] = !DILocation(line: 93, column: 9, scope: [[DBG52]])
// CHECK: [[META62]] = !DILocalVariable(name: "b", scope: [[DBG52]], file: [[META7]], line: 93, type: [[META15]])
// CHECK: [[META63]] = !DILocation(line: 93, column: 12, scope: [[DBG52]])
// CHECK: [[DBG64]] = !DILocation(line: 93, column: 17, scope: [[DBG52]])
// CHECK: [[DBG65]] = !DILocation(line: 94, column: 1, scope: [[DBG52]])
// CHECK: [[DBG66]] = distinct !DISubprogram(name: "fS4", linkageName: "_Z3fS4v", scope: [[META7]], file: [[META7]], line: 115, type: [[META8]], scopeLine: 115, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META67:![0-9]+]])
// CHECK: [[META67]] = !{[[META68]]}
// CHECK: [[META68]] = !DILocalVariable(name: "s4", scope: [[DBG66]], file: [[META7]], line: 116, type: [[META69:![0-9]+]])
// CHECK: [[META69]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S4", file: [[META7]], line: 96, size: 64, flags: DIFlagTypePassByValue, elements: [[META70:![0-9]+]], identifier: "_ZTS2S4")
// CHECK: [[META70]] = !{[[META71:![0-9]+]], [[META72:![0-9]+]], [[META73:![0-9]+]]}
// CHECK: [[META71]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META69]], file: [[META7]], line: 97, baseType: [[META15]], size: 16, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META72]] = !DIDerivedType(tag: DW_TAG_member, scope: [[META69]], file: [[META7]], line: 98, baseType: [[META15]], offset: 32, flags: DIFlagBitField, extraData: i64 32)
// CHECK: [[META73]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META69]], file: [[META7]], line: 99, baseType: [[META15]], size: 16, offset: 32, flags: DIFlagBitField, extraData: i64 32)
// CHECK: [[META74]] = !DILocation(line: 116, column: 6, scope: [[DBG66]])
// CHECK: [[META75]] = !DILocalVariable(name: "a", scope: [[DBG66]], file: [[META7]], line: 117, type: [[META15]])
// CHECK: [[META76]] = !DILocation(line: 117, column: 9, scope: [[DBG66]])
// CHECK: [[META77]] = !DILocalVariable(name: "b", scope: [[DBG66]], file: [[META7]], line: 117, type: [[META15]])
// CHECK: [[META78]] = !DILocation(line: 117, column: 12, scope: [[DBG66]])
// CHECK: [[DBG79]] = !DILocation(line: 117, column: 17, scope: [[DBG66]])
// CHECK: [[DBG80]] = !DILocation(line: 118, column: 1, scope: [[DBG66]])
// CHECK: [[DBG81]] = distinct !DISubprogram(name: "fS5", linkageName: "_Z3fS5v", scope: [[META7]], file: [[META7]], line: 140, type: [[META8]], scopeLine: 140, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META82:![0-9]+]])
// CHECK: [[META82]] = !{[[META83]]}
// CHECK: [[META83]] = !DILocalVariable(name: "s5", scope: [[DBG81]], file: [[META7]], line: 141, type: [[META84:![0-9]+]])
// CHECK: [[META84]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S5", file: [[META7]], line: 123, size: 32, flags: DIFlagTypePassByValue, elements: [[META85:![0-9]+]], identifier: "_ZTS2S5")
// CHECK: [[META85]] = !{[[META86:![0-9]+]], [[META87:![0-9]+]]}
// CHECK: [[META86]] = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: [[META84]], file: [[META7]], line: 124, baseType: [[META15]], size: 15, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META87]] = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: [[META84]], file: [[META7]], line: 125, baseType: [[META15]], size: 16, offset: 15, flags: DIFlagBitField, extraData: i64 0)
// CHECK: [[META88]] = !DILocation(line: 141, column: 6, scope: [[DBG81]])
// CHECK: [[META89]] = !DILocalVariable(name: "a", scope: [[DBG81]], file: [[META7]], line: 142, type: [[META15]])
// CHECK: [[META90]] = !DILocation(line: 142, column: 9, scope: [[DBG81]])
// CHECK: [[DBG91]] = !DILocation(line: 142, column: 17, scope: [[DBG81]])
// CHECK: [[DBG92]] = !DILocation(line: 143, column: 1, scope: [[DBG81]])
//.
