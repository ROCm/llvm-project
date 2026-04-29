// COM: Test v_cvt_f32_fp8 CLAMP=1 (E5M3) UE5M3->F32 unpack conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_f32_fp8
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: byte extraction, NaN detection, exp-31
// COM: detection, exp-31 direct F32 construction, F16 base path, exp-31
// COM: select, and NaN override.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=1 byte_sel=0 patch — original site has s_branch, and
// COM: the trampoline contains the full unpack conversion sequence.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE0 %s

// BYTE0-LABEL: <test_cvt_f32_fp8_byte0>:
// BYTE0:       s_branch
// COM: --- Byte extraction (byte_sel=0: v_and_b32) ---
// BYTE0:       v_and_b32{{.*}}0xff, v1
// COM: --- NaN detection ---
// BYTE0-NEXT:  v_cmp_eq_u32{{.*}}0xff
// BYTE0-NEXT:  s_mov_b32
// COM: --- Exp-31 detection ---
// BYTE0-NEXT:  v_cmp_lt_u32{{.*}}0xf7
// BYTE0-NEXT:  s_mov_b32
// COM: --- Exp-31 direct F32 construction ---
// BYTE0-NEXT:  v_and_b32{{.*}} 7,
// BYTE0-NEXT:  v_lshlrev_b32{{.*}}, 20
// BYTE0-NEXT:  v_or_b32{{.*}}0x47800000
// COM: --- F16 base path ---
// BYTE0-NEXT:  v_lshlrev_b32{{.*}}, 7
// BYTE0-NEXT:  v_cvt_f32_f16
// COM: --- Exp-31 select ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_mov_b32{{.*}}0x7fa3d000
// BYTE0-NEXT:  v_cndmask_b32{{.*}}v0,

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=1 byte_sel=2 patch (v_bfe_u32 extraction).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE2 %s

// BYTE2-LABEL: <test_cvt_f32_fp8_byte2>:
// BYTE2:       s_branch
// COM: --- Byte extraction (byte_sel=2: v_bfe_u32) ---
// BYTE2:       v_bfe_u32{{.*}}v6, 16, 8
// COM: --- NaN detection ---
// BYTE2-NEXT:  v_cmp_eq_u32{{.*}}0xff
// BYTE2-NEXT:  s_mov_b32
// COM: --- Exp-31 detection ---
// BYTE2-NEXT:  v_cmp_lt_u32{{.*}}0xf7
// BYTE2-NEXT:  s_mov_b32
// COM: --- Exp-31 direct F32 construction ---
// BYTE2-NEXT:  v_and_b32{{.*}} 7,
// BYTE2-NEXT:  v_lshlrev_b32{{.*}}, 20
// BYTE2-NEXT:  v_or_b32{{.*}}0x47800000
// COM: --- F16 base path ---
// BYTE2-NEXT:  v_lshlrev_b32{{.*}}, 7
// BYTE2-NEXT:  v_cvt_f32_f16
// COM: --- Exp-31 select ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_mov_b32{{.*}}0x7fa3d000
// BYTE2-NEXT:  v_cndmask_b32{{.*}}v5,

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=0 (E4M3) instruction is NOT patched.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOCLAMP %s

// NOCLAMP-LABEL: <test_cvt_f32_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_f32_fp8

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Kernel 1: CLAMP=1, byte_sel=0 (should be patched) ---
.globl test_cvt_f32_fp8_byte0
.p2align 8
.type test_cvt_f32_fp8_byte0,@function
test_cvt_f32_fp8_byte0:
  v_cvt_f32_fp8 v0, v1 clamp
  s_endpgm
.Ltest_cvt_f32_fp8_byte0_end:
.size test_cvt_f32_fp8_byte0, .Ltest_cvt_f32_fp8_byte0_end-test_cvt_f32_fp8_byte0

// --- Kernel 2: CLAMP=1, byte_sel=2 (raw encoding for OPSEL[1:0]) ---
.globl test_cvt_f32_fp8_byte2
.p2align 8
.type test_cvt_f32_fp8_byte2,@function
test_cvt_f32_fp8_byte2:
  // v_cvt_f32_fp8 v5, v6 clamp byte_sel=2
  // dword0 = 0xD5EC8805 (CLAMP=1, OPSEL[0]=1 -> byte_sel=2, vdst=v5)
  // dword1 = 0x02010106 (src0=v6, no modifiers)
  .long 0xD5EC8805
  .long 0x02010106
  s_endpgm
.Ltest_cvt_f32_fp8_byte2_end:
.size test_cvt_f32_fp8_byte2, .Ltest_cvt_f32_fp8_byte2_end-test_cvt_f32_fp8_byte2

// --- Kernel 3: no clamp (should NOT be patched) ---
.globl test_cvt_f32_fp8_noclamp
.p2align 8
.type test_cvt_f32_fp8_noclamp,@function
test_cvt_f32_fp8_noclamp:
  v_cvt_f32_fp8 v10, v11
  s_endpgm
.Ltest_cvt_f32_fp8_noclamp_end:
.size test_cvt_f32_fp8_noclamp, .Ltest_cvt_f32_fp8_noclamp_end-test_cvt_f32_fp8_noclamp

.rodata
.p2align 8
.amdhsa_kernel test_cvt_f32_fp8_byte0
  .amdhsa_next_free_vgpr 2
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte2
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_noclamp
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
