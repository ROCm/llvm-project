; RUN: llc -O0 -verify-machineinstrs -stop-after=regallocfast %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-dwarfdump --debug-info - | FileCheck %s

; Empty and fragment-only direct expressions describe the spill slot itself.
; Computed locations and indirect DBG_VALUEs need an explicit spill load.
; Stack-value expressions get their spill load during frame-index lowering.

; MIR-DAG: ![[LEN:[0-9]+]] = !DILocalVariable(name: ".str123.len"
; MIR-DAG: ![[STR:[0-9]+]] = !DILocalVariable(name: "str123"
; MIR-DAG: ![[FRAGMENT:[0-9]+]] = !DILocalVariable(name: "fragment"
; MIR-DAG: ![[COMPUTED:[0-9]+]] = !DILocalVariable(name: "computed"
; MIR-DAG: ![[IMPLICIT:[0-9]+]] = !DILocalVariable(name: "implicit"
; MIR: body: |
; MIR-DAG: DBG_VALUE %stack.[[SLOT:[0-9]+]], 0, ![[LEN]], !DIExpression(),
; MIR-DAG: DBG_VALUE %stack.[[SLOT]], 0, ![[FRAGMENT]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; MIR-DAG: DBG_VALUE %stack.[[SLOT]], 0, ![[COMPUTED]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 1),
; MIR-DAG: DBG_VALUE %stack.[[SLOT]], 0, ![[IMPLICIT]], !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value),
; MIR-DAG: DBG_VALUE %stack.{{[0-9]+}}, 0, ![[STR]], !DIExpression(DW_OP_deref),

; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_location (DW_OP_fbreg {{[+-][0-9]+}}, DW_OP_deref)
; CHECK: DW_AT_name ("str123")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg [[OFFSET:[+-][0-9]+]])
; CHECK-NEXT: DW_AT_name (".str123.len")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg [[OFFSET]], DW_OP_piece 0x4)
; CHECK-NEXT: DW_AT_name ("fragment")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg [[OFFSET]], DW_OP_deref, DW_OP_plus_uconst 0x1)
; CHECK-NEXT: DW_AT_name ("computed")
; CHECK: DW_TAG_variable
; CHECK: DW_AT_location (DW_OP_fbreg [[OFFSET]], DW_OP_deref_size 0x8, DW_OP_plus_uconst 0x1, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("implicit")

target triple = "x86_64-unknown-linux-gnu"

define void @show_str(ptr %str, i64 %len) #0 !dbg !5 {
entry:
  #dbg_value(i64 %len, !9, !DIExpression(), !10)
  #dbg_value(i64 %len, !15, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !10)
  #dbg_value(i64 %len, !16, !DIExpression(DW_OP_plus_uconst, 1), !10)
  #dbg_value(i64 %len, !17, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value), !10)
  #dbg_declare(ptr %str, !11, !DIExpression(), !10)
  call void @clobber(), !dbg !13
  call void @use(ptr %str, i64 %len), !dbg !13
  ret void, !dbg !13
}

declare void @use(ptr, i64)
declare void @clobber()

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !1, producer: "flang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.f90", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "show_str", linkageName: "show_str_", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DISubroutineType(cc: DW_CC_normal, types: !7)
!7 = !{null, !14}
!9 = !DILocalVariable(name: ".str123.len", scope: !5, file: !1, type: !12, flags: DIFlagArtificial)
!10 = !DILocation(line: 2, column: 24, scope: !5)
!11 = !DILocalVariable(name: "str123", arg: 1, scope: !5, file: !1, line: 2, type: !14)
!12 = !DIBasicType(name: "integer(kind=8)", size: 64, encoding: DW_ATE_signed)
!13 = !DILocation(line: 4, column: 1, scope: !5)
!14 = !DIStringType(stringLength: !9, encoding: DW_ATE_ASCII)
!15 = !DILocalVariable(name: "fragment", scope: !5, file: !1, type: !12)
!16 = !DILocalVariable(name: "computed", scope: !5, file: !1, type: !12)
!17 = !DILocalVariable(name: "implicit", scope: !5, file: !1, type: !12)
