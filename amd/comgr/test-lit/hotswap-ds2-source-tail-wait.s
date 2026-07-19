// COM: An 8-byte DS2 source leaves one dword after its branch. Keep only the
// COM: exact final s_wait_dscnt 0 from the split replacement in that dword, so
// COM: the sled needs two 8-byte DS stores plus one 4-byte branch-back: exactly
// COM: 20 bytes instead of 24. The sled returns to source+4, executes the wait,
// COM: then falls through to the original continuation.

// RUN: %clang -x assembler-with-cpp -DPROTECTED=0 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <source_tail_wait>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: ds_store_b64 v2, v[0:1] offset:256
// DISASM-NEXT: ds_store_b64 v2, v[4:5] offset:768
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: A callable entry at source+4 is strictly inside every replacement
// COM: window for the eight-byte DS2. It must reject the rewrite rather than
// COM: silently becoming branch-tail padding in any fallback placement.
// RUN: %clang -x assembler-with-cpp -DPROTECTED=1 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.protected.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.protected.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=PROTECTED %s
// PROTECTED: replacement source [0x0, 0x8) contains protected interior entry 0x4
// PROTECTED: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl source_tail_wait
.type source_tail_wait,@function
source_tail_wait:
  ds_store_2addr_b64 v2, v[0:1], v[4:5] offset0:32 offset1:96
#if PROTECTED
.globl protected_interior
.type protected_interior,@function
.set protected_interior, source_tail_wait + 4
#endif
  s_endpgm

// This is deliberately the exact compact-layout capacity. The legacy layout
// needs 24 bytes and therefore cannot use it. Keep the sled inside the source
// function so it is legitimate replacement-body storage.
.Lexact_20_byte_sled:
.rept 5
  s_nop 0
.endr
.size source_tail_wait, .-source_tail_wait

.rodata
.p2align 8
.amdhsa_kernel source_tail_wait
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: source_tail_wait
      .symbol: source_tail_wait.kd
      .sgpr_count: 2
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
