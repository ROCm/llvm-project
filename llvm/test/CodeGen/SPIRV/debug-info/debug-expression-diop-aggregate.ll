; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s

; HIP-style DIOp locations on a struct and an array local. DIOpDeref names the
; struct and array types the locals' storage already declares.
; No spirv-val run: NonSemantic.Shader.DebugInfo.100 only allows OpConstant
; operands in DebugOperation, so spirv-val rejects a type id there.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[POINT:%[0-9]+]] = OpTypeStruct [[FLOAT]] [[FLOAT]]
; CHECK-DAG: [[ARR:%[0-9]+]] = OpTypeArray [[I32T]] {{%[0-9]+}}
; CHECK-DAG: [[CDEREF:%[0-9]+]] = OpConstant [[I32T]] 10014{{ *$}}
; CHECK-DAG: [[ARG:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}}{{ *$}}
; CHECK-DAG: [[DEREFP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[POINT]]{{ *$}}
; CHECK-DAG: [[DEREFA:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[ARR]]{{ *$}}
; CHECK-DAG: [[EXPRP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREFP]]{{ *$}}
; CHECK-DAG: [[EXPRA:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREFA]]{{ *$}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPRP]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPRA]]

target triple = "spirv64-amd-amdhsa"

%struct.Point = type { float, float }

define spir_func void @f() !dbg !5 {
entry:
  %p = alloca %struct.Point, align 4
  %arr = alloca [4 x i32], align 4
    #dbg_declare(ptr %p, !9, !DIExpression(DIOpArg(0, ptr), DIOpDeref(%struct.Point)), !14)
    #dbg_declare(ptr %arr, !10, !DIExpression(DIOpArg(0, ptr), DIOpDeref([4 x i32])), !14)
  store float 1.0, ptr %p, align 4, !dbg !14
  store i32 1, ptr %arr, align 4, !dbg !14
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-diop-aggregate.cpp", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !DILocalVariable(name: "p", scope: !5, file: !1, line: 2, type: !11)
!10 = !DILocalVariable(name: "arr", scope: !5, file: !1, line: 3, type: !13)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Point", file: !1, line: 1, size: 64, elements: !12)
!12 = !{!15, !16}
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !17)
!14 = !DILocation(line: 4, column: 1, scope: !5)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !11, file: !1, line: 1, baseType: !8, size: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !1, line: 1, baseType: !8, size: 32, offset: 32)
!17 = !{!18}
!18 = !DISubrange(count: 4)
