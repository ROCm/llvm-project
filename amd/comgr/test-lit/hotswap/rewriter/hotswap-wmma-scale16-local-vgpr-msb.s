// Verify that a Scale16 lowering recovers its incoming VGPR-MSB mode from the
// local straight-line setter when object-wide CFG recovery is unavailable.
//
// The register-target call gives the object an unresolved control-flow target,
// which declines the whole-function mode analysis. The Scale16 is preceded
// immediately by its own s_set_vgpr_msb, so its original operands already
// depend on that setter and the local scan may supply the mode.
//
// hotswap-wmma-scale16-local-vgpr-msb-reject.s is the matching negative: the
// same unresolved call with no local setter must fail closed. The pair is what
// proves the local scan, rather than the CFG analysis, supplied the mode here.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// COM: Object-wide mode recovery is declined ...
// API: unresolved call target
// COM: ... yet the required lowering still applies.
// API: wmma_scale16: exact K-split
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_scale16_local_mode>:
// DISASM-NOT: v_wmma_scale16
// COM: The recovered mode must be the locally established 0x5, so matrix A is
// COM: gathered from bank 1. A mode-zero fallback would read v16 instead of
// COM: v272 and silently feed the replacements the wrong matrix.
// DISASM: v_mov_b32_e32 v8, v16{{.*v272.*}}
// COM: Both replacement passes consume the masked low-bank A and the gathered
// COM: even/odd scale pairs. Matrix B's incoming src1 bank already matches the
// COM: scratch bank, so both passes read the original v[288:295] in place
// COM: rather than an above-KD copy of it.
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[32:39]{{.*v\[288:295\].*}}v16, v17
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[32:39]{{.*v\[288:295\].*}}v18, v19

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_scale16_local_mode
.p2align 8
.type test_scale16_local_mode,@function
test_scale16_local_mode:
  s_set_vgpr_msb 0x5
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v[48:49], v[50:51] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
  s_swap_pc_i64 s[30:31], s[0:1]
  s_endpgm
.Ltest_scale16_local_mode_end:
.size test_scale16_local_mode, .Ltest_scale16_local_mode_end-test_scale16_local_mode

.rodata
.p2align 8
.amdhsa_kernel test_scale16_local_mode
  .amdhsa_next_free_vgpr 304
  .amdhsa_next_free_sgpr 32
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_scale16_local_mode
      .symbol: test_scale16_local_mode.kd
      .sgpr_count: 32
      .vgpr_count: 304
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
