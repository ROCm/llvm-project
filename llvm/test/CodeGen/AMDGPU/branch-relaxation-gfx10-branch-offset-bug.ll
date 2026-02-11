; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -amdgpu-late-wave-transform=1 -amdgpu-s-branch-bits=7 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX1030 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -amdgpu-late-wave-transform=1 -amdgpu-s-branch-bits=7 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX1010 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-late-wave-transform=1 -amdgpu-s-branch-bits=7 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX1030 %s

; With the Late Wave Transform, the gfx1010-specific 0x3f offset
; distinction does not apply due to uniform branches lowering through
; VCC rather than SCC. All targets produce the same long-branch
; relaxation pattern.

; GCN-LABEL: long_forward_scc_branch_3f_offset_bug:
; GFX1030: s_cmp_lg_u32
; GFX1030: s_cbranch_vccz
; GFX1030: s_getpc_b64
; GFX1030: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ([[ENDBB:.LBB[0-9]+_[0-9]+]]-

; GFX1010: s_cmp_lg_u32
; GFX1010: s_cbranch_vccz  [[LOOP_BB:.LBB[0-9]+_[0-9]+]]
; GFX1010: s_getpc_b64
; GFX1010-NEXT: [[POST_GETPC:.Lpost_getpc[0-9]+]]:{{$}}
; GFX1010-NEXT: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ([[ENDBB:.LBB[0-9]+_[0-9]+]]-[[POST_GETPC]])&4294967295
; GFX1010-NEXT: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ([[ENDBB:.LBB[0-9]+_[0-9]+]]-[[POST_GETPC]])>>32
; GFX1010: [[LOOP_BB]]:

; GCN: v_nop
; GCN: s_sleep
; GCN: s_cbranch_vccz

; GCN: [[ENDBB]]:
; GCN: global_store_{{dword|b32}}
define amdgpu_kernel void @long_forward_scc_branch_3f_offset_bug(ptr addrspace(1) %arg, i32 %cnd0) #0 {
bb0:
  %cmp0 = icmp eq i32 %cnd0, 0
  br i1 %cmp0, label %bb2, label %bb3

bb2:
  %val = call i32 asm sideeffect
   "s_mov_b32 $0, 0
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", "=s"()   ; 20 * 12 = 240
  call void @llvm.amdgcn.s.sleep(i32 0) ; +4 = 244
  %cmp1 = icmp eq i32 %val, 0           ; +4 = 248
  br i1 %cmp1, label %bb2, label %bb3   ; +4 (gfx1030), +8 with workaround (gfx1010)

bb3:
  store volatile i32 %cnd0, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}long_forward_exec_branch_3f_offset_bug:
; GFX1030: v_cmp_ne_u32
; GFX1030: s_mov_b32 exec_lo
; GFX1030: s_cbranch_execnz [[TAKEN_BB:.LBB[0-9]+_[0-9]+]]

; GFX1010: v_cmp_ne_u32
; GFX1010: s_mov_b32 exec_lo
; GFX1010: s_cbranch_execnz  [[TAKEN_BB:.LBB[0-9]+_[0-9]+]]

; GCN: s_getpc_b64
; GCN-NEXT: [[POST_GETPC:.Lpost_getpc[0-9]+]]:{{$}}
; GCN-NEXT: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, ([[ENDBB:.LBB[0-9]+_[0-9]+]]-[[POST_GETPC]])&4294967295
; GCN-NEXT: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, ([[ENDBB]]-[[POST_GETPC]])>>32
; GCN: [[TAKEN_BB]]:

; GCN: v_nop
; GCN: s_sleep
; GCN: s_cbranch_execz

; GCN: [[ENDBB]]:
; GCN: global_store_{{dword|b32}}
define void @long_forward_exec_branch_3f_offset_bug(ptr addrspace(1) %arg, i32 %cnd0) #0 {
bb0:
  %cmp0 = icmp eq i32 %cnd0, 0
  br i1 %cmp0, label %bb2, label %bb3

bb2:
  %val = call i32 asm sideeffect
   "v_mov_b32 $0, 0
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", "=v"()   ; 20 * 12 = 240
  call void @llvm.amdgcn.s.sleep(i32 0) ; +4 = 244
  %cmp1 = icmp eq i32 %val, 0           ; +4 = 248
  br i1 %cmp1, label %bb2, label %bb3   ; +4 (gfx1030), +8 with workaround (gfx1010)

bb3:
  store volatile i32 %cnd0, ptr addrspace(1) %arg
  ret void
}

declare void @llvm.amdgcn.s.sleep(i32 immarg)
