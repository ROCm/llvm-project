// COM: Transactional displacement repairs the carry-chain PC materialization
// COM: used by production rocSOLVER objects. The growing DS rewrite moves the
// COM: in-.text target, so the low/high add immediates must be re-encoded.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: transactional displacement: collected 1 growing edit(s)
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <transactional_addcarry>:
// DISASM:       s_get_pc_i64 s[0:1]
// DISASM-NEXT:  s_add_co_u32 s0, s0, {{(0x[0-9a-f]+|[0-9]+)}}
// DISASM-NEXT:  s_add_co_ci_u32 s1, s1, {{(lit\(0x[0-9a-f]+\)|0x[0-9a-f]+|[0-9]+)}}
// DISASM-NEXT:  s_set_pc_i64 s[0:1]
// DISASM-NOT:   ds_load_2addr_stride64_b32
// DISASM:       ds_load_b32 v4, v2 offset:256
// DISASM-NEXT:  ds_load_b32 v5, v2 offset:768
// DISASM:       s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl transactional_addcarry
.p2align 8
.type transactional_addcarry,@function
transactional_addcarry:
  // s_get_pc_i64 captures the address of the following low-word add. The
  // original target is 20 bytes after that PC base.
  s_get_pc_i64 s[0:1]
  s_add_co_u32 s0, s0, 20
  s_add_co_ci_u32 s1, s1, 0
  s_set_pc_i64 s[0:1]
  ds_load_2addr_stride64_b32 v[4:5], v2 offset0:1 offset1:3
.Ltarget:
  s_endpgm
.Lend:
.size transactional_addcarry, .Lend-transactional_addcarry

.rodata
.p2align 8
.amdhsa_kernel transactional_addcarry
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: transactional_addcarry
      .symbol: transactional_addcarry.kd
      .sgpr_count: 2
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 256
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
.end_amdgpu_metadata
