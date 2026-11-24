; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; DebugOperation under NonSemantic.Shader.DebugInfo.100 only allows OpConstant
; i32 operands, so a type id here does not pass spirv-val. 

; HIP-style DIOp locations on scalar, struct and array locals. DebugOperation
; opcodes follow the numbering SPIRV-LLVM-Translator expects. Each type operand is a type the locals' storage already
; declares; the ptr is the untyped pointer type of the alloca's storage class.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[POINT:%[0-9]+]] = OpTypeStruct [[FLOAT]] [[FLOAT]]
; CHECK-DAG: [[ARR:%[0-9]+]] = OpTypeArray [[I32T]] {{%[0-9]+}}
; CHECK-DAG: [[PTR:%[0-9]+]] = OpTypeUntypedPointerKHR Function
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[CARG:%[0-9]+]] = OpConstant [[I32T]] 10003{{ *$}}
; CHECK-DAG: [[CDEREF:%[0-9]+]] = OpConstant [[I32T]] 10014{{ *$}}
; CHECK-DAG: [[ARG:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CARG]] [[C0]] [[PTR]]{{ *$}}
; CHECK-DAG: [[DEREFX:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[I32T]]{{ *$}}
; CHECK-DAG: [[DEREFP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[POINT]]{{ *$}}
; CHECK-DAG: [[DEREFA:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[ARR]]{{ *$}}
; CHECK-DAG: [[EXPRX:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREFX]]{{ *$}}
; CHECK-DAG: [[EXPRP:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREFP]]{{ *$}}
; CHECK-DAG: [[EXPRA:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREFA]]{{ *$}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPRX]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPRP]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPRA]]

target triple = "spirv64-amd-amdhsa"

%struct.Point = type { float, float }

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
  %p = alloca %struct.Point, align 4
  %arr = alloca [4 x i32], align 4
    #dbg_declare(ptr %x, !9, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), !14)
    #dbg_declare(ptr %p, !10, !DIExpression(DIOpArg(0, ptr), DIOpDeref(%struct.Point)), !14)
    #dbg_declare(ptr %arr, !11, !DIExpression(DIOpArg(0, ptr), DIOpDeref([4 x i32])), !14)
  store i32 1, ptr %x, align 4, !dbg !14
  store float 1.0, ptr %p, align 4, !dbg !14
  store i32 1, ptr %arr, align 4, !dbg !14
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-diop.cpp", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocalVariable(name: "p", scope: !5, file: !1, line: 3, type: !12)
!11 = !DILocalVariable(name: "arr", scope: !5, file: !1, line: 4, type: !13)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Point", file: !1, line: 1, size: 64, elements: !15)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !18)
!14 = !DILocation(line: 5, column: 1, scope: !5)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, file: !1, line: 1, baseType: !8, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !12, file: !1, line: 1, baseType: !8, size: 32, offset: 32)
!18 = !{!19}
!19 = !DISubrange(count: 4)
