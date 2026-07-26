// COM: A reachable backedge into function entry does not prove that the
// COM: descriptor mask is zero. Entry is an unknown root unless executing the
// COM: entry instruction itself establishes an accepted zero definition.
// COM: Without that rule, every node in this cycle has a predecessor and the
// COM: backward search can terminate without finding any zero definition.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=LOG %s
// LOG-NOT: descriptor workgroup_mask is already zero on every path
// LOG: tensor_load_to_lds: s4 live, save/restore via s12
// LOG: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DIS %s
// DIS-LABEL: <test_tensor_mask_entry_backedge>:
// DIS: s_cbranch_scc0
// DIS-NEXT: s_branch
// DIS: s_mov_b32 s12, s4
// DIS-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DIS-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DIS-NEXT: s_mov_b32 s4, s12
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_entry_backedge
.p2align 8
.type test_tensor_mask_entry_backedge,@function
test_tensor_mask_entry_backedge:
  s_mov_b32 s0, s4
  s_cbranch_scc0 .Ltensor
  s_branch test_tensor_mask_entry_backedge
.Ltensor:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
  .rept 8
    s_nop 0
  .endr
.Ltest_tensor_mask_entry_backedge_end:
.size test_tensor_mask_entry_backedge, .Ltest_tensor_mask_entry_backedge_end-test_tensor_mask_entry_backedge

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_entry_backedge
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_entry_backedge
      .symbol: test_tensor_mask_entry_backedge.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
