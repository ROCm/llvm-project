// COM: A single A0 tag is not an object-wide target-state certificate. Mixed
// COM: kernel metadata must run the rewrite, retag every kernel, and only then
// COM: permit a byte-identical repeated request.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=FIRST %s
// FIRST-NOT: every kernel already reports gfx1250 revision A0
// FIRST: applied 1 instruction patches
// FIRST: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: ds_load_2addr_b64
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s
// METADATA-NOT: .gfx1250_revision: B0
// METADATA-COUNT-2: .gfx1250_revision: A0
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out2.elf 2>&1 | %FileCheck --check-prefix=SECOND %s
// SECOND: every kernel already reports gfx1250 revision A0
// SECOND: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl already_a0
.p2align 8
.type already_a0,@function
already_a0:
  s_endpgm
.size already_a0, .-already_a0

.globl still_b0
.p2align 8
.type still_b0,@function
still_b0:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_endpgm
  .rept 16
    s_nop 0
  .endr
.size still_b0, .-still_b0

.rodata
.p2align 8
.amdhsa_kernel already_a0
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel still_b0
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: already_a0
      .symbol: already_a0.kd
      .gfx1250_revision: A0
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: still_b0
      .symbol: still_b0.kd
      .gfx1250_revision: B0
      .sgpr_count: 2
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
