// COM: Two adjacent 8-byte DS2 instructions coalesce into one local sled. The
// COM: combined replacements are 40 bytes; moving only the final four-byte
// COM: drain to source+4 leaves 36 body bytes plus one four-byte branch-back,
// COM: exactly filling this 40-byte sled. Preselection and emission must use
// COM: the same compact-layout size.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: hotswap: coalesced 2 adjacent ds_2addr rewrites
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <coalesced_source_tail_wait>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: ds_store_b64 v2, v[0:1] offset:256
// DISASM-NEXT: ds_store_b64 v2, v[4:5] offset:768
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_store_b64 v8, v[6:7] offset:512
// DISASM-NEXT: ds_store_b64 v8, v[10:11] offset:1024
// DISASM-NEXT: s_branch {{.*}} <coalesced_source_tail_wait+0x4>

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl coalesced_source_tail_wait
.type coalesced_source_tail_wait,@function
coalesced_source_tail_wait:
  ds_store_2addr_b64 v2, v[0:1], v[4:5] offset0:32 offset1:96
  ds_store_2addr_b64 v8, v[6:7], v[10:11] offset0:64 offset1:128
  s_endpgm

.Lexact_40_byte_sled:
.rept 10
  s_nop 0
.endr
.size coalesced_source_tail_wait, .-coalesced_source_tail_wait

.rodata
.p2align 8
.amdhsa_kernel coalesced_source_tail_wait
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: coalesced_source_tail_wait
      .symbol: coalesced_source_tail_wait.kd
      .sgpr_count: 2
      .vgpr_count: 12
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
