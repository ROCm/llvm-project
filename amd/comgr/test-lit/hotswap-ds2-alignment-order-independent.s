// COM: The later DS2 address is defined before an earlier DS2 that requires
// COM: expansion. Alignment exemptions must be cached before patching: once
// COM: the earlier instruction is relabelled <replaced>, it must not hide the
// COM: aligned definition from the later site.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: hotswap: ds_2addr: preserved proven-aligned ds_load_2addr_b64 at 0x
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <order_independent_alignment>:
// DISASM-NEXT: s_set_vgpr_msb 0x400
// DISASM-NEXT: v_add_nc_u32_e64 v6, 0x100, 0
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_load_2addr_b64 v[12:15], v6 offset1:1
// DISASM-NEXT: s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl order_independent_alignment
.type order_independent_alignment,@function
order_independent_alignment:
  s_set_vgpr_msb 0x400
  v_add_nc_u32_e64 v6, 0x100, 0
  ds_load_2addr_b64 v[8:11], v7 offset1:1
  ds_load_2addr_b64 v[12:15], v6 offset1:1
  s_endpgm

// The first expansion exactly fills this compact 20-byte sled.
.Lexact_20_byte_sled:
.rept 5
  s_nop 0
.endr
.size order_independent_alignment, .-order_independent_alignment

.rodata
.p2align 8
.amdhsa_kernel order_independent_alignment
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: order_independent_alignment
      .symbol: order_independent_alignment.kd
      .sgpr_count: 2
      .vgpr_count: 16
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
