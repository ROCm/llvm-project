// COM: If whole-object displacement rejects the input, the speculative run is
// COM: discarded and the growing patch is emitted through the established
// COM: trampoline path from the original object.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: transactional displacement: collected 1 growing edit(s)
// LOG: transactional displacement declined: debug/unwind section '.debug_info'
// LOG: retrying the original object with trampoline placement
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <transactional_fallback>:
// DISASM-NOT:   ds_load_2addr_stride64_b32
// DISASM:       s_branch
// DISASM:       s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out2.elf
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl transactional_fallback
.p2align 8
.type transactional_fallback,@function
transactional_fallback:
  ds_load_2addr_stride64_b32 v[4:5], v2 offset0:1 offset1:3
  s_endpgm
.Lend:
.size transactional_fallback, .Lend-transactional_fallback

.rodata
.p2align 8
.amdhsa_kernel transactional_fallback
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel

.section .debug_info,"",@progbits
.byte 0

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: transactional_fallback
      .symbol: transactional_fallback.kd
      .sgpr_count: 1
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 256
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
.end_amdgpu_metadata
