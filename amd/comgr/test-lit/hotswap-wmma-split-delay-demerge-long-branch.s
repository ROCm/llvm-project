// Combined-delay demerge is limited to a short trampoline. A far source would
// replace the delay window with generated set-PC control flow that tensor-mask
// idempotence does not treat as a straight-line local definition. Fail before
// mutating the predecessor or writing a partially rewritten output.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: rm -f %t.out.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: test ! -e %t.out.elf
// API: WMMA split: combined-delay demerge at 0x{{[0-9A-F]+}} requires a short trampoline
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_delay_demerge_far
.p2align 8
.type test_wmma_delay_demerge_far,@function
test_wmma_delay_demerge_far:
  v_readfirstlane_b32 s4, v40
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_wmma_delay_demerge_far_end:
.size test_wmma_delay_demerge_far, .Ltest_wmma_delay_demerge_far_end-test_wmma_delay_demerge_far

// Keep the trampoline pool beyond s_branch reach without extending the
// function or creating another gateway sled.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel test_wmma_delay_demerge_far
  .amdhsa_next_free_vgpr 41
  .amdhsa_next_free_sgpr 28
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_delay_demerge_far
      .symbol: test_wmma_delay_demerge_far.kd
      .sgpr_count: 28
      .vgpr_count: 41
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
