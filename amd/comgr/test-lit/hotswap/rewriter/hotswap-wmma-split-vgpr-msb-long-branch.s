// A far crossing split needs a forward gateway. Source-window growth may move
// the callable-entry VGPR-MSB setter only as the first copied instruction, so
// every entry still establishes the original mode before either WMMA half.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// DISASM-LABEL: <test_wmma_vgpr_msb_far>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_endpgm
// DISASM:      s_set_vgpr_msb 0
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[162:169], v[250:257], v[114:121], 0
// DISASM-NEXT: s_set_vgpr_msb 1
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[162:169], v[2:9]{{.*v\[258:265\].*}}, v[122:129], v[162:169]
// DISASM-NEXT: s_set_vgpr_msb 0x100

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_vgpr_msb_far
.p2align 8
.type test_wmma_vgpr_msb_far,@function
test_wmma_vgpr_msb_far:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_fp8_fp8 v[162:169], v[250:265], v[114:129], 0
  s_endpgm
.Ltest_wmma_vgpr_msb_far_end:
.size test_wmma_vgpr_msb_far, .Ltest_wmma_vgpr_msb_far_end-test_wmma_vgpr_msb_far

// Safe external gateway space after the no-fallthrough terminator.
.rept 8
  s_nop 0
.endr

// Keep the appended pool beyond s_branch reach from the split site.
.rept 40000
  s_mov_b32 s3, s4
.endr

.rodata
.p2align 8
.amdhsa_kernel test_wmma_vgpr_msb_far
  .amdhsa_next_free_vgpr 266
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_vgpr_msb_far
      .symbol: test_wmma_vgpr_msb_far.kd
      .sgpr_count: 0
      .vgpr_count: 266
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
