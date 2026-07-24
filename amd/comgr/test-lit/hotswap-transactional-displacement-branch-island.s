// COM: Transactional displacement relaxes a conditional branch whose target
// COM: moves just beyond simm16 range. A guarded 256-byte island preserves
// COM: ordinary fallthrough and all later kernel-entry alignment.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: displacement: planned guarded branch island
// LOG: displacement: relaxed branch
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <transactional_branch_island>:
// DISASM-NEXT:  s_cbranch_execz 16416
// DISASM:       s_branch 63
// DISASM-NEXT:  s_branch 16417
// DISASM-NOT:   ds_load_2addr_stride64_b32
// DISASM:       ds_load_b32 v4, v2 offset:256
// DISASM-NEXT:  ds_load_b32 v5, v2 offset:768
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl transactional_branch_island
.p2align 8
.type transactional_branch_island,@function
transactional_branch_island:
  // The original target is exactly +32767 dwords from the branch PC. Growing
  // the DS instruction between the branch and target requires relaxation.
  s_cbranch_execz .Ltarget
  .rept 32765
    s_nop 0
  .endr
  ds_load_2addr_stride64_b32 v[4:5], v2 offset0:1 offset1:3
.Ltarget:
  s_endpgm
.Lend:
.size transactional_branch_island, .Lend-transactional_branch_island

.rodata
.p2align 8
.amdhsa_kernel transactional_branch_island
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: transactional_branch_island
      .symbol: transactional_branch_island.kd
      .sgpr_count: 1
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 256
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
.end_amdgpu_metadata
