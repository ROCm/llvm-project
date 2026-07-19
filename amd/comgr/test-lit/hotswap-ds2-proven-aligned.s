// COM: A0 accepts an ordinary DS_READ2/DS_WRITE2 when the address is aligned.
// COM: HotSwap may preserve that compact form only when must-dataflow proves
// COM: VALU-dst and DS-src0 select the same VGPR bank and a same-block constant
// COM: definition remains valid under unchanged MODE and EXEC.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <aligned_entry>:
// DISASM: v_add_nc_u32_e64 v4, 0x4000, 0
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_restored_zero>:
// DISASM: s_set_vgpr_msb 0x400
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_high_bank>:
// DISASM: s_set_vgpr_msb 0x41
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_store>:
// DISASM: ds_store_2addr_b64
// DISASM-LABEL: <aligned_sgpr_mov>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_dual_slot0>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_dual_slot1>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_long_mul>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_immediate_mov>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_mask_loop>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_empty_bypass>:
// DISASM: ds_load_2addr_b64
// DISASM-LABEL: <aligned_standard_return>:
// DISASM: ds_load_2addr_b64
// DISASM: s_set_pc_i64 s[30:31]

// DISASM-LABEL: <unequal_mode>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <mode_changed_after_def>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <exec_changed_after_def>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <unequal_merge>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <entry_after_def>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <address_redefined>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <misaligned_constant>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <misaligned_immediate_mov>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <stride_form>:
// DISASM-NOT: ds_load_2addr_stride64_b64
// DISASM-LABEL: <sgpr_redefined>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <sgpr_bypass>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <dual_other_slot_unaligned>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <pair_mask_clobber>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <preexisting_mask_restore>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <active_bypass>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <loop_address_redefined>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <call_clobber>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM-LABEL: <long_mode_mismatch>:
// DISASM-NOT: ds_load_2addr_b64

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro tail
  s_endpgm
  .rept 8
    s_nop 0
  .endr
.endm

.macro begin_func name
  .globl \name
  .p2align 8
  .type \name,@function
\name:
.endm

.macro end_func name
  .size \name, .-\name
.endm

begin_func aligned_entry
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_entry

begin_func aligned_restored_zero
  s_set_vgpr_msb 0x400
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_restored_zero

begin_func aligned_high_bank
  s_set_vgpr_msb 0x41
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_high_bank

begin_func aligned_store
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:1 offset1:2
  tail
end_func aligned_store

begin_func aligned_sgpr_mov
  s_mul_i32 s4, s4, 56
  v_mov_b32 v4, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_sgpr_mov

begin_func aligned_dual_slot0
  s_mul_i32 s4, s4, 56
  v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v5, s5
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_dual_slot0

begin_func aligned_dual_slot1
  s_mul_i32 s5, s5, 56
  v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v5, s5
  ds_load_2addr_b64 v[0:3], v5 offset0:1 offset1:2
  tail
end_func aligned_dual_slot1

begin_func aligned_long_mul
  s_set_vgpr_msb 0x40
  v_mul_lo_u32 v4, v0, 56
  s_and_saveexec_b32 s4, s5
  s_or_b32 exec_lo, exec_lo, s4
  s_set_vgpr_msb 0x1
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_long_mul

begin_func aligned_immediate_mov
  v_mov_b32 v4, 0x43a8
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_immediate_mov

begin_func aligned_mask_loop
  s_mov_b32 s4, 0
.Lmask_loop:
  v_mul_lo_u32 v4, v0, 56
  ds_load_b32 v1, v4 offset:36
  s_wait_dscnt 0
  v_cmp_ne_u32_e32 vcc_lo, 0, v1
  s_or_b32 s4, vcc_lo, s4
  s_and_not1_b32 exec_lo, exec_lo, s4
  s_cbranch_execnz .Lmask_loop
  s_or_b32 exec_lo, exec_lo, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_mask_loop

begin_func aligned_empty_bypass
  s_cbranch_scc1 .Lempty_candidate
  s_mov_b32 exec_lo, 0
  s_branch .Lempty_join
.Lempty_candidate:
  v_mul_lo_u32 v4, v0, 56
.Lempty_join:
  s_nop 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func aligned_empty_bypass

begin_func aligned_standard_return
  v_mul_lo_u32 v4, v0, 56
  s_and_saveexec_b32 s4, s5
  s_or_b32 exec_lo, exec_lo, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_set_pc_i64 s[30:31]
end_func aligned_standard_return

begin_func unequal_mode
  s_set_vgpr_msb 0x40
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func unequal_mode

begin_func mode_changed_after_def
  v_add_nc_u32_e64 v4, 0x4000, 0
  s_set_vgpr_msb 0x41
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func mode_changed_after_def

begin_func exec_changed_after_def
  v_add_nc_u32_e64 v4, 0x4000, 0
  s_mov_b32 exec_lo, -1
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func exec_changed_after_def

begin_func unequal_merge
  s_cbranch_scc1 .Lunequal
  s_set_vgpr_msb 0x41
  s_branch .Lmerge
.Lunequal:
  s_set_vgpr_msb 0x40
.Lmerge:
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func unequal_merge

begin_func entry_after_def
  v_add_nc_u32_e64 v4, 0x4000, 0
  s_cbranch_scc1 .Lentry
  s_nop 0
.Lentry:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func entry_after_def

begin_func address_redefined
  v_add_nc_u32_e64 v4, 0x4000, 0
  v_mov_b32 v4, v5
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func address_redefined

begin_func misaligned_constant
  v_add_nc_u32_e64 v4, 0x4001, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func misaligned_constant

begin_func misaligned_immediate_mov
  v_mov_b32 v4, 0x43a9
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func misaligned_immediate_mov

begin_func stride_form
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func stride_form

begin_func sgpr_redefined
  s_mul_i32 s4, s4, 56
  s_mov_b32 s4, s5
  v_mov_b32 v4, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func sgpr_redefined

begin_func sgpr_bypass
  s_cbranch_scc1 .Lsgpr_copy
  s_mul_i32 s4, s4, 56
.Lsgpr_copy:
  v_mov_b32 v4, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func sgpr_bypass

begin_func dual_other_slot_unaligned
  s_mul_i32 s4, s4, 56
  v_dual_mov_b32 v4, s4 :: v_dual_mov_b32 v5, s5
  ds_load_2addr_b64 v[0:3], v5 offset0:1 offset1:2
  tail
end_func dual_other_slot_unaligned

begin_func pair_mask_clobber
  v_mul_lo_u32 v4, v0, 56
  s_mov_b32 s4, exec_lo
  s_mov_b64 s[4:5], -1
  s_mov_b32 exec_lo, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func pair_mask_clobber

begin_func preexisting_mask_restore
  s_mov_b32 s4, exec_lo
  v_mul_lo_u32 v4, v0, 56
  s_mov_b32 exec_lo, s4
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func preexisting_mask_restore

begin_func active_bypass
  s_cbranch_scc1 .Lactive_candidate
  s_nop 0
  s_branch .Lactive_join
.Lactive_candidate:
  v_mul_lo_u32 v4, v0, 56
.Lactive_join:
  s_nop 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func active_bypass

begin_func loop_address_redefined
  s_branch .Laddress_candidate
.Laddress_clobber:
  v_mov_b32 v4, v5
  s_branch .Laddress_use
.Laddress_candidate:
  v_mul_lo_u32 v4, v0, 56
  s_cbranch_scc1 .Laddress_clobber
.Laddress_use:
  s_nop 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func loop_address_redefined

begin_func call_clobber
  v_mul_lo_u32 v4, v0, 56
  s_call_i64 s[30:31], .Lcall_target
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
.Lcall_target:
  s_set_pc_i64 s[30:31]
end_func call_clobber

begin_func long_mode_mismatch
  s_set_vgpr_msb 0x40
  v_mul_lo_u32 v4, v0, 56
  s_and_saveexec_b32 s4, s5
  s_or_b32 exec_lo, exec_lo, s4
  s_set_vgpr_msb 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  tail
end_func long_mode_mismatch
