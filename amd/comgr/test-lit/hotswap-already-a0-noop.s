// COM: Uniform per-kernel A0 metadata is an object-wide target-state
// COM: certificate. A repeated B0-to-A0 request must return the object
// COM: unchanged, even when its instructions would match B0 rewrite rules if
// COM: reinterpreted without that metadata.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: every kernel already reports gfx1250 revision A0
// LOG-NOT: ds_2addr:
// LOG: RESULT: SUCCESS
// RUN: cmp %t.elf %t.out.elf
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: ds_load_2addr_b64

// COM: The target-state certificate suppresses only B0-to-A0 instruction
// COM: patches. An independently requested entry trampoline still runs.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --entry-trampolines --output %t.entry.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=ENTRY %s
// ENTRY: every kernel already reports gfx1250 revision A0
// ENTRY: RESULT: SUCCESS
// RUN: %llvm-readelf -s %t.entry.elf | %FileCheck --check-prefix=ENTRY-SYM %s
// ENTRY-SYM-NOT: already_a0.stub
// RUN: %llvm-objdump -d %t.entry.elf | \
// RUN:   %FileCheck --check-prefix=ENTRY-DISASM %s
// ENTRY-DISASM-LABEL: <already_a0>:
// ENTRY-DISASM-NEXT: global_wb
// ENTRY-DISASM-NEXT: v_nop
// ENTRY-DISASM-NEXT: ds_load_2addr_b64

// COM: Direct entry displacement is stepping-neutral. A later B0-to-A0 request
// COM: takes the A0 metadata fast path and returns the object unchanged.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.entry.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.entry.repeat.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=ENTRY-REPEAT %s
// ENTRY-REPEAT: every kernel already reports gfx1250 revision A0
// ENTRY-REPEAT: RESULT: SUCCESS
// RUN: cmp %t.entry.elf %t.entry.repeat.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl already_a0
.p2align 8
.type already_a0,@function
already_a0:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_endpgm
.size already_a0, .-already_a0

.rodata
.p2align 8
.amdhsa_kernel already_a0
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
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
