// COM: Discover a DS2's combined-delay owner from encoded target geometry.
// COM: Interior positions are supported at every legal skip distance, while
// COM: either protected target and overlapping owners remain conservative.

// RUN: %clang -x assembler-with-cpp -DMIN_SKIP -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.min.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.min.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.min.out.elf 2>&1 | %FileCheck --check-prefix=MIN %s
// MIN-NOT: hotswap: error:
// MIN: hotswap: ds_2addr: demerged combined s_delay_alu at 0x
// MIN-NOT: hotswap: error:
// MIN: RESULT: SUCCESS

// The DS2 is four instructions after its owner, rather than at the formerly
// assumed owner+2 position, and remains strictly between both targets.
// RUN: %clang -x assembler-with-cpp -DDEEP_INTERIOR \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.deep.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.deep.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.deep.out.elf 2>&1 | %FileCheck --check-prefix=DEEP %s
// DEEP-NOT: hotswap: error:
// DEEP: hotswap: ds_2addr: demerged combined s_delay_alu at 0x
// DEEP-NOT: hotswap: error:
// DEEP: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.deep.out.elf | %FileCheck --check-prefix=DEEP-DISASM %s
// DEEP-DISASM-LABEL: <combined_delay_ownership>:
// DEEP-DISASM-NOT: ds_load_2addr_b64
// DEEP-DISASM: s_delay_alu instid0(VALU_DEP_1)
// DEEP-DISASM-NEXT: v_cmp_eq_u32_e64 s12, 0, v6
// DEEP-DISASM: s_delay_alu instid0(SALU_CYCLE_1)
// DEEP-DISASM: ds_load_b64 v[2:3], v52 offset:1224
// DEEP-DISASM-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// DEEP-DISASM-NOT: ds_load_2addr_b64

// RUN: %clang -x assembler-with-cpp -DFIRST_TARGET \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.first.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.first.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=FIRST %s
// FIRST: hotswap: error: ds_2addr: protected source at 0x4 has no supported combined-delay window
// FIRST: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DSECOND_TARGET \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.second.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.second.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=SECOND %s
// SECOND: hotswap: error: ds_2addr: protected source at 0x14 has no supported combined-delay window
// SECOND: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DOVERLAPPING_OWNERS \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.overlap.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.overlap.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=OVERLAP %s
// OVERLAP: hotswap: error: ds_2addr: protected source at 0x10 has no supported
// OVERLAP-SAME: combined-delay window
// OVERLAP: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl combined_delay_ownership
.p2align 8
.type combined_delay_ownership,@function
combined_delay_ownership:
#if defined(MIN_SKIP)
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  v_cmp_eq_u32_e64 s12, 0, v6
#elif defined(DEEP_INTERIOR)
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  s_wait_dscnt 0
  s_nop 0
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_wait_dscnt 1
  v_cmp_eq_u32_e64 s12, 0, v6
#elif defined(FIRST_TARGET)
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_wait_dscnt 0
  s_nop 0
  s_wait_dscnt 1
  v_cmp_eq_u32_e64 s12, 0, v6
#elif defined(SECOND_TARGET)
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  s_wait_dscnt 0
  s_nop 0
  s_wait_dscnt 1
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
#elif defined(OVERLAPPING_OWNERS)
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  s_wait_dscnt 0
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_wait_dscnt 1
  v_cmp_eq_u32_e64 s12, 0, v6
#endif
  s_endpgm
  .rept 48
    s_nop 0
  .endr
.Lcombined_delay_ownership_end:
.size combined_delay_ownership, .Lcombined_delay_ownership_end-combined_delay_ownership

.rodata
.p2align 8
.amdhsa_kernel combined_delay_ownership
  .amdhsa_next_free_vgpr 199
  .amdhsa_next_free_sgpr 13
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: combined_delay_ownership
      .symbol: combined_delay_ownership.kd
      .sgpr_count: 13
      .vgpr_count: 199
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
