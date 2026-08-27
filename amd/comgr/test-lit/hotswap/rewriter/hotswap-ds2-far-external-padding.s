// COM: A pair-backed-only object receives no correctness benefit from filling
// COM: audited external padding with local DS2 bodies. Preserve every slot for
// COM: routing unless maximum matching actually places a registerless site.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: exposed
// LOG-SAME: unowned unreachable external padding sled(s) for local DS2 bodies
// LOG-SAME: preserving 20 routing bytes per run
// LOG: hotswap: preserved 5 audited slot(s) for routing because maximum matching placed no registerless DS2 site
// LOG-NOT: hotswap: placed pair-backed DS2 site
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_ds2_far_external_padding>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_get_pc_i64
// DISASM: ds_load_b32 v0, v2 offset:4
// DISASM-NEXT: ds_load_b32 v1, v2 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out2.elf 2>&1 | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_far_external_padding
.p2align 8
.type test_ds2_far_external_padding,@function
test_ds2_far_external_padding:
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.size test_ds2_far_external_padding, .-test_ds2_far_external_padding

// Zero decodes as s_code_end. The preceding s_endpgm proves this complete run
// unreachable by fallthrough; no branch/call target enters it.
.fill 32, 4, 0

// Keep a hypothetical appended trampoline pool outside signed s_branch reach.
.rept 40000
  s_mov_b32 s2, s3
.endr

.rodata
.p2align 8
.amdhsa_kernel test_ds2_far_external_padding
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2_far_external_padding
      .symbol: test_ds2_far_external_padding.kd
      .sgpr_count: 4
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
