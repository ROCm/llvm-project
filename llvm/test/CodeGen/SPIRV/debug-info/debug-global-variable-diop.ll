; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; No spirv-val run: NonSemantic.Shader.DebugInfo.100 has no operand 10 in
; DebugGlobalVariable, and DebugOperation only allows OpConstant operands.

; A global with a DIOp expression gets it as DebugGlobalVariable operand 10,
; after a DebugInfoNone Static Member Declaration, as SPIRV-LLVM-Translator
; does. A global with an ordinary expression keeps nine operands.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[NAMES:%[0-9]+]] = OpString "s"
; CHECK-DAG: [[NAMEN:%[0-9]+]] = OpString "n"
; CHECK-DAG: [[EMPTY:%[0-9]+]] = OpString ""
; CHECK-DAG: OpName [[S:%[0-9]+]] "s"
; CHECK-DAG: OpName [[N:%[0-9]+]] "n"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[WGPTR:%[0-9]+]] = OpTypeUntypedPointerKHR Workgroup
; CHECK-DAG: [[S]] = OpUntypedVariableKHR [[WGPTR]] Workgroup [[I32T]]
; CHECK-DAG: [[N]] = OpUntypedVariableKHR {{%[0-9]+}} CrossWorkgroup [[I32T]]
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C1:%[0-9]+]] = OpConstant [[I32T]] 1{{ *$}}
; CHECK-DAG: [[C2:%[0-9]+]] = OpConstant [[I32T]] 2{{ *$}}
; CHECK-DAG: [[LOCALDEF:%[0-9]+]] = OpConstant [[I32T]] 12{{ *$}}
; CHECK-DAG: [[DEF:%[0-9]+]] = OpConstant [[I32T]] 8{{ *$}}
; CHECK-DAG: [[CARG:%[0-9]+]] = OpConstant [[I32T]] 10003{{ *$}}
; CHECK-DAG: [[CDEREF:%[0-9]+]] = OpConstant [[I32T]] 10014{{ *$}}
; CHECK-DAG: [[SRC:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit
; CHECK-DAG: [[INT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic
; CHECK-DAG: [[NONE:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugInfoNone
; CHECK-DAG: [[ARG:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CARG]] [[C0]] [[WGPTR]]{{ *$}}
; CHECK-DAG: [[DEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[I32T]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREF]]{{ *$}}

; Name, Type, Source, Line, Column, Parent, Linkage Name, Variable, Flags,
; Static Member Declaration, DIOp expression.
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAMES]] [[INT]] [[SRC]] [[C1]] [[C0]] [[CU]] [[EMPTY]] [[S]] [[LOCALDEF]] [[NONE]] [[EXPR]]{{ *$}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugGlobalVariable [[NAMEN]] [[INT]] [[SRC]] [[C2]] [[C0]] [[CU]] [[EMPTY]] [[N]] [[DEF]]{{ *$}}

target triple = "spirv64-amd-amdhsa"

@s = internal addrspace(3) global i32 poison, align 4, !dbg !0
@n = addrspace(1) global i32 0, align 4, !dbg !5

define spir_kernel void @k() !dbg !10 {
entry:
  store i32 1, ptr addrspace(3) @s, align 4, !dbg !13
  store i32 2, ptr addrspace(1) @n, align 4, !dbg !13
  ret void, !dbg !13
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DIOpArg(0, ptr addrspace(3)), DIOpDeref(i32)))
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 1, type: !4, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !7, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "debug-global-variable-diop.cpp", directory: "/src")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "n", scope: !2, file: !3, line: 2, type: !4, isLocal: false, isDefinition: true)
!7 = !{!0, !5}
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "k", linkageName: "k", scope: !3, file: !3, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 4, column: 1, scope: !10)
