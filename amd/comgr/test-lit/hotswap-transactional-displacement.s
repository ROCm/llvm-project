// COM: Multiple growing replacements are committed through one displacement
// COM: transaction. Expanded instructions stay on the straight-line path and
// COM: a second rewrite is byte-identical.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: transactional displacement: collected 2 growing edit(s)
// LOG: displacement: grew ELF
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <transactional_ds2>:
// DISASM-NOT:   ds_load_2addr_stride64_b32
// DISASM-NOT:   s_branch
// DISASM:       ds_load_b32 v4, v2 offset:256
// DISASM-NEXT:  ds_load_b32 v5, v2 offset:768
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  ds_load_b32 v6, v3 offset:512
// DISASM-NEXT:  ds_load_b32 v7, v3 offset:1024
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out2.elf
// RUN: cmp %t.out.elf %t.out2.elf

// COM: A long-lived process samples the displacement switch per request.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-displacement-toggle --output %t.toggle.elf \
// RUN:   | %FileCheck --check-prefix=TOGGLE %s
// TOGGLE: DISPLACEMENT_TOGGLE: OBSERVED
// TOGGLE: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.text
.globl transactional_ds2
.p2align 8
.type transactional_ds2,@function
transactional_ds2:
  ds_load_2addr_stride64_b32 v[4:5], v2 offset0:1 offset1:3
  ds_load_2addr_stride64_b32 v[6:7], v3 offset0:2 offset1:4
  s_endpgm
.Lend:
.size transactional_ds2, .Lend-transactional_ds2

.rodata
.p2align 8
.amdhsa_kernel transactional_ds2
  .amdhsa_wavefront_size32 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 1
  .amdhsa_group_segment_fixed_size 256
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .gfx1250_revision: B0
      .name: transactional_ds2
      .symbol: transactional_ds2.kd
      .sgpr_count: 1
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .kernarg_segment_align: 8
      .group_segment_fixed_size: 256
      .private_segment_fixed_size: 0
      .max_flat_workgroup_size: 64
      .wavefront_size: 32
.end_amdgpu_metadata
