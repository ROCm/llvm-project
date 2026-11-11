; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; DebugOperation under NonSemantic.Shader.DebugInfo.100 only allows OpConstant
; i32 operands, so a type id here does not pass spirv-val. .200 is the set
; whose grammar accepts these operands.

; A HIP-style DIOp location on a local. Opcodes follow SPIRV-LLVM-Translator:
; DIOp_Begin is 10001, DIOpArg's bitcode id is 2, DIOpDeref's is 13.
; The pointer and i32 types are ones the function already uses. The ptr is the
; untyped pointer type of the alloca's storage class.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PTR:%[0-9]+]] = OpTypeUntypedPointerKHR Function
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[CARG:%[0-9]+]] = OpConstant [[I32T]] 10003{{ *$}}
; CHECK-DAG: [[CDEREF:%[0-9]+]] = OpConstant [[I32T]] 10014{{ *$}}
; CHECK-DAG: [[ARG:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CARG]] [[C0]] [[PTR]]{{ *$}}
; CHECK-DAG: [[DEREF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugOperation [[CDEREF]] [[I32T]]{{ *$}}
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression [[ARG]] [[DEREF]]{{ *$}}
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugDeclare {{%[0-9]+}} {{%[0-9]+}} [[EXPR]]

target triple = "spirv64-amd-amdhsa"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), !10)
  store i32 1, ptr %x, align 4, !dbg !10
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-expression-diop.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
