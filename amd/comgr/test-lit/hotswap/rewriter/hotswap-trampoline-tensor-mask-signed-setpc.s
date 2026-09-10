// COM: Recognize only Tensile's exact contiguous signed-direction reusable-PC
// COM: transfer. Both arithmetic arms compute the same local target. A direct
// COM: entry into the positive arm invalidates that target proof and forces
// COM: the tensor's at-site fallback.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOCAL %s
// LOCAL: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// LOCAL-NOT: tensor_load_to_lds: s4
// LOCAL: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DIS %s
// DIS: s_and_b32 s4, s4, 0xfff70000
// DIS-NOT: s_pack_hh_b32_b16
// DIS: tensor_load_to_lds s[0:3], s[4:11]

// RUN: sed 's/^\.set alternate_arm_entry, 0$/.set alternate_arm_entry, 1/' \
// RUN:   %s > %t.alternate.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.alternate.s -o %t.alternate.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.alternate.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.alternate.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=ALTERNATE %s
// ALTERNATE: tensor CFG rejected alternate entry into reusable-PC sequence
// ALTERNATE-NOT: cleared workgroup_mask at descriptor definition
// ALTERNATE: tensor_load_to_lds: s4 dead, no save/restore needed
// ALTERNATE: RESULT: SUCCESS

.set alternate_arm_entry, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_signed_setpc
.p2align 8
.type test_tensor_mask_signed_setpc,@function
test_tensor_mask_signed_setpc:
  s_and_b32 s4, s4, 0xfff7ffff
  s_cmp_eq_u32 s0, s0
.if alternate_arm_entry
  s_cbranch_scc0 .Lpositive
.endif
  s_cbranch_execz .Ltensor
.Lgetpc:
  s_get_pc_i64 s[70:71]
  s_add_co_i32 s72, .Ltarget-(.Lgetpc+4), 0
  s_cmp_ge_i32 s72, 0
  s_cbranch_scc1 .Lpositive
  s_abs_i32 s72, s72
  s_sub_co_u32 s70, s70, s72
  s_sub_co_ci_u32 s71, s71, 0
  s_set_pc_i64 s[70:71]
.Lpositive:
  s_add_co_u32 s70, s70, s72
  s_add_co_ci_u32 s71, s71, 0
  s_set_pc_i64 s[70:71]
.Ltensor:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s4, 0
  s_cmp_eq_u32 s0, s0
  s_endpgm
.Ltarget:
  s_mov_b32 s4, 0
  s_cmp_eq_u32 s0, s0
  s_endpgm
.Ltest_tensor_mask_signed_setpc_end:
.size test_tensor_mask_signed_setpc, .Ltest_tensor_mask_signed_setpc_end-test_tensor_mask_signed_setpc

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_signed_setpc
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 73
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_signed_setpc
      .symbol: test_tensor_mask_signed_setpc.kd
      .sgpr_count: 73
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
