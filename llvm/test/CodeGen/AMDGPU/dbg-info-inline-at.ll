; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 3
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck %s

define amdgpu_kernel void @_Z3fooPiiii(ptr addrspace(1) nocapture noundef writeonly %c.coerce, i32 noundef %a, i32 noundef %b, i32 noundef %d) !dbg !9 {
; CHECK-LABEL: _Z3fooPiiii:
; CHECK:       .Lfunc_begin0:
; CHECK-NEXT:    .file 0 "test" "a.hip" md5 0x004a28df8cfd98cdd2c71d5d814d9c6b
; CHECK-NEXT:    .cfi_sections .debug_frame
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:  ; %bb.0: ; %entry
; CHECK-NEXT:    .cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ;
; CHECK-NEXT:    .cfi_undefined 16
; CHECK-NEXT:    .file 1 "." "a.h"
; CHECK-NEXT:    .loc 1 5 12 prologue_end ; ./a.h:5:12 @[ a.hip:12:8 ]
; CHECK-NEXT:    s_load_dwordx4 s[0:3], s[8:9], 0x8
; CHECK-NEXT:    s_load_dwordx2 s[4:5], s[8:9], 0x0
; CHECK-NEXT:    v_mov_b32_e32 v0, 0
; CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_add_i32 s1, s1, s0
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:    .loc 1 5 12 is_stmt 0 ; ./a.h:5:12 @[ a.hip:13:9 ]
; CHECK-NEXT:    s_add_i32 s0, s2, s0
; CHECK-NEXT:  .Ltmp1:
; CHECK-NEXT:    .file 2 "a.hip"
; CHECK-NEXT:    .loc 2 13 6 is_stmt 1 ; a.hip:13:6
; CHECK-NEXT:    s_mul_i32 s0, s0, s1
; CHECK-NEXT:    v_mov_b32_e32 v1, s0
; CHECK-NEXT:    global_store_dword v0, v1, s[4:5]
; CHECK-NEXT:    .loc 2 14 1 ; a.hip:14:1
; CHECK-NEXT:    s_endpgm
; CHECK-NEXT:  .Ltmp2:
entry:
  %add.i = add nsw i32 %b, %a, !dbg !13
  %add.i3 = add nsw i32 %d, %a, !dbg !17
  %mul = mul nsw i32 %add.i3, %add.i, !dbg !19
  store i32 %mul, ptr addrspace(1) %c.coerce, align 4, !dbg !19, !tbaa !20
  ret void, !dbg !24
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.hip", directory: "test", checksumkind: CSK_MD5, checksum: "004a28df8cfd98cdd2c71d5d814d9c6b")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{!"clang version 20.0.0"}
!9 = distinct !DISubprogram(name: "foo", scope: !10, file: !10, line: 11, type: !11, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DIFile(filename: "a.hip", directory: "test")
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 12, scope: !14, inlinedAt: !16)
!14 = distinct !DISubprogram(name: "bar", scope: !15, file: !15, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DIFile(filename: "./a.h", directory: "test")
!16 = distinct !DILocation(line: 12, column: 8, scope: !9)
!17 = !DILocation(line: 5, column: 12, scope: !14, inlinedAt: !18)
!18 = distinct !DILocation(line: 13, column: 9, scope: !9)
!19 = !DILocation(line: 13, column: 6, scope: !9)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !DILocation(line: 14, column: 1, scope: !9)
