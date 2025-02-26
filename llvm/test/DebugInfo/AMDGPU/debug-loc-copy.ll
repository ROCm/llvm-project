; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck -check-prefix=GCN %s

; Verify that the debug locations in this function are correct, in particular
; that the location for %cast doesn't appear in the block of %lab.

define void @_Z12lane_pc_testj() #0 !dbg !9 {
; GCN-LABEL: _Z12lane_pc_testj:
; GCN:       .Lfunc_begin0:
; GCN-NEXT:    .file 0 "/" "t.cpp"
; GCN-NEXT:    .loc 0 3 0 ; t.cpp:3:0
; GCN-NEXT:    .cfi_sections .debug_frame
; GCN-NEXT:    .cfi_startproc
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    .cfi_llvm_def_aspace_cfa 64, 0, 6
; GCN-NEXT:    .cfi_llvm_register_pair 16, 62, 32, 63, 32
; GCN-NEXT:    .cfi_undefined 1536
; GCN-NEXT:    .cfi_undefined 1537
; GCN-NEXT:    .cfi_undefined 1538
; GCN-NEXT:    .cfi_undefined 36
; GCN-NEXT:    .cfi_undefined 37
; GCN-NEXT:    .cfi_undefined 38
; GCN-NEXT:    .cfi_undefined 39
; GCN-NEXT:    .cfi_undefined 40
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:  ; %bb.1: ; %lab
; GCN-NEXT:  .Ltmp0:
; GCN-NEXT:    .loc 0 12 1 prologue_end ; t.cpp:12:1
; GCN-NEXT:    s_mov_b64 s[4:5], src_private_base
; GCN-NEXT:    s_mov_b32 s6, 32
; GCN-NEXT:    s_lshr_b64 s[4:5], s[4:5], s6
; GCN-NEXT:    s_mov_b64 s[6:7], 0
; GCN-NEXT:    s_mov_b32 s5, -1
; GCN-NEXT:    s_lshr_b32 s8, s32, 5
; GCN-NEXT:    s_cmp_lg_u32 s8, s5
; GCN-NEXT:    s_cselect_b32 s5, s4, s7
; GCN-NEXT:    s_cselect_b32 s4, s8, s6
; GCN-NEXT:    v_mov_b32_e32 v2, 0
; GCN-NEXT:    .loc 0 13 1 ; t.cpp:13:1
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    v_mov_b32_e32 v1, s5
; GCN-NEXT:    flat_store_dword v[0:1], v2
; GCN-NEXT:    v_mov_b32_e32 v2, 1
; GCN-NEXT:    .loc 0 14 1 ; t.cpp:14:1
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    v_mov_b32_e32 v1, s5
; GCN-NEXT:    flat_store_dword v[0:1], v2
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN-NEXT:  .Ltmp1:
  %alloc = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %alloc to ptr, !dbg !12
  br label %lab

lab:
  store i32 0, ptr %cast, align 4, !dbg !13
  store i32 1, ptr %cast, align 4, !dbg !14
  ret void
}

attributes #0 = { noinline optnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = distinct !DISubprogram(name: "lane_pc_test", linkageName: "_Z12lane_pc_testj", scope: !1, file: !1, line: 1, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, type: !10, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 12, column: 1, scope: !9)
!13 = !DILocation(line: 13, column: 1, scope: !9)
!14 = !DILocation(line: 14, column: 1, scope: !9)
