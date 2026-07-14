// An atomic combined-delay relocation must not consume another instruction
// that still needs a per-instruction HotSwap pass. Exercise both discovery
// orders: DS2 before WMMA and WMMA before DS2. Both rewrites must fail before
// installing a source branch, rather than marking the later site replaced or
// claimed and silently skipping its required correction.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.ds2-first.elf
// RUN: rm -f %t.ds2-first.out.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.ds2-first.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.ds2-first.out.elf --expect-status ERROR \
// RUN:   2>&1 | %FileCheck --check-prefix=DS2-FIRST %s
// RUN: test ! -e %t.ds2-first.out.elf
// DS2-FIRST: ds_2addr: combined-delay window member at 0x{{[0-9A-F]+}} requires a separate HotSwap patch
// DS2-FIRST: ds_2addr: protected source at 0x{{[0-9A-F]+}} has no supported combined-delay window
// DS2-FIRST: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.wmma-first.elf
// RUN: rm -f %t.wmma-first.out.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.wmma-first.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.wmma-first.out.elf --expect-status ERROR \
// RUN:   2>&1 | %FileCheck --check-prefix=WMMA-FIRST %s
// RUN: test ! -e %t.wmma-first.out.elf
// WMMA-FIRST: WMMA split: delay-window member at 0x{{[0-9A-F]+}} requires a separate HotSwap patch
// WMMA-FIRST: WMMA split: protected site at 0x{{[0-9A-F]+}} would suppress another required HotSwap patch
// WMMA-FIRST: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl combined_delay_required_patch_ownership
.p2align 8
.type combined_delay_required_patch_ownership,@function
combined_delay_required_patch_ownership:
#if CASE == 1
  // DS2 is strictly between both delay targets. Its demerge used to relocate
  // and mark the second-target WMMA replaced before the split pass saw it.
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
  s_nop 0
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
#elif CASE == 2
  // WMMA is the first target and DS2 is an interior member. Its split used to
  // claim the DS2's linked address, causing the outer dispatcher to skip it.
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_nop 0
  s_nop 0
#else
  .error "CASE must select a test body"
#endif
  s_endpgm
.Lcombined_delay_required_patch_ownership_end:
.size combined_delay_required_patch_ownership, .Lcombined_delay_required_patch_ownership_end-combined_delay_required_patch_ownership

.rodata
.p2align 8
.amdhsa_kernel combined_delay_required_patch_ownership
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: combined_delay_required_patch_ownership
      .symbol: combined_delay_required_patch_ownership.kd
      .sgpr_count: 2
      .vgpr_count: 64
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
