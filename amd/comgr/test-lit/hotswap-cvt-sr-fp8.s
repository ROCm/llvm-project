// COM: Test v_cvt_sr_fp8_f32 CLAMP=1 (E5M3) stochastic-round conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_sr_fp8_f32
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: NaN detection, stochastic noise injection,
// COM: F32->F16->UE5M3 conversion, overflow clamping, NaN override, and byte merge.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=1 byte_sel=0 patch — original site has s_branch, and
// COM: the trampoline contains the full SR conversion sequence.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE0 %s

// BYTE0-LABEL: <test_cvt_sr_fp8_byte0>:
// BYTE0:       s_branch
// COM: --- NaN detection ---
// BYTE0:       v_and_b32{{.*}}0x7fffffff, v1
// BYTE0-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// BYTE0-NEXT:  s_mov_b32
// COM: --- Clamp negative ---
// BYTE0-NEXT:  v_max_num_f32{{.*}}, 0, v1
// COM: --- Stochastic noise injection ---
// BYTE0-NEXT:  v_and_b32{{.*}}0x7fffff
// BYTE0-NEXT:  v_lshrrev_b32{{.*}}, 12, v2
// BYTE0-NEXT:  v_add
// BYTE0-NEXT:  v_and_b32{{.*}}0x7fffff
// BYTE0-NEXT:  v_max_num_f32{{.*}}, 0, v1
// BYTE0-NEXT:  v_bfi_b32
// COM: --- F32 -> F16 -> UE5M3 ---
// BYTE0-NEXT:  v_cvt_f16_f32
// BYTE0-NEXT:  v_lshrrev_b32
// COM: --- Overflow clamp ---
// BYTE0-NEXT:  v_min_u32
// COM: --- NaN override ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_mov_b32
// BYTE0-NEXT:  v_cndmask_b32
// COM: --- Byte merge (byte_sel=0: single bfi) ---
// BYTE0-NEXT:  v_bfi_b32 v0,

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=1 byte_sel=2 patch (shift + bfi merge).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE2 %s

// BYTE2-LABEL: <test_cvt_sr_fp8_byte2>:
// BYTE2:       s_branch
// COM: --- NaN detection ---
// BYTE2:       v_and_b32{{.*}}0x7fffffff, v6
// BYTE2-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// BYTE2-NEXT:  s_mov_b32
// COM: --- Clamp negative ---
// BYTE2-NEXT:  v_max_num_f32{{.*}}, 0, v6
// COM: --- Stochastic noise injection ---
// BYTE2-NEXT:  v_and_b32{{.*}}0x7fffff
// BYTE2-NEXT:  v_lshrrev_b32{{.*}}, 12, v7
// BYTE2-NEXT:  v_add
// BYTE2-NEXT:  v_and_b32{{.*}}0x7fffff
// BYTE2-NEXT:  v_max_num_f32{{.*}}, 0, v6
// BYTE2-NEXT:  v_bfi_b32
// COM: --- F32 -> F16 -> UE5M3 ---
// BYTE2-NEXT:  v_cvt_f16_f32
// BYTE2-NEXT:  v_lshrrev_b32
// COM: --- Overflow clamp ---
// BYTE2-NEXT:  v_min_u32
// COM: --- NaN override ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_mov_b32
// BYTE2-NEXT:  v_cndmask_b32
// COM: --- Byte merge (byte_sel=2: shift + bfi) ---
// BYTE2-NEXT:  v_lshlrev_b32
// BYTE2-NEXT:  v_bfi_b32 v5,

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=0 (E4M3) instruction is NOT patched.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOCLAMP %s

// NOCLAMP-LABEL: <test_cvt_sr_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_sr_fp8_f32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Kernel 1: CLAMP=1, byte_sel=0 (should be patched) ---
.globl test_cvt_sr_fp8_byte0
.p2align 8
.type test_cvt_sr_fp8_byte0,@function
test_cvt_sr_fp8_byte0:
  v_cvt_sr_fp8_f32 v0, v1, v2 clamp
  s_endpgm
.Ltest_cvt_sr_fp8_byte0_end:
.size test_cvt_sr_fp8_byte0, .Ltest_cvt_sr_fp8_byte0_end-test_cvt_sr_fp8_byte0

// --- Kernel 2: CLAMP=1, byte_sel=2 (raw encoding for OPSEL[3:2]) ---
.globl test_cvt_sr_fp8_byte2
.p2align 8
.type test_cvt_sr_fp8_byte2,@function
test_cvt_sr_fp8_byte2:
  // v_cvt_sr_fp8_f32 v5, v6, v7 clamp byte_sel=2
  // dword0 = 0xD76BC005 (CLAMP=1, OPSEL[3]=1, OPSEL[2]=0 -> byte_sel=2, vdst=v5)
  // dword1 = 0x02020F06 (src0=v6, src1=v7, no modifiers)
  .long 0xD76BC005
  .long 0x02020F06
  s_endpgm
.Ltest_cvt_sr_fp8_byte2_end:
.size test_cvt_sr_fp8_byte2, .Ltest_cvt_sr_fp8_byte2_end-test_cvt_sr_fp8_byte2

// --- Kernel 3: no clamp (should NOT be patched) ---
.globl test_cvt_sr_fp8_noclamp
.p2align 8
.type test_cvt_sr_fp8_noclamp,@function
test_cvt_sr_fp8_noclamp:
  v_cvt_sr_fp8_f32 v10, v11, v12
  s_endpgm
.Ltest_cvt_sr_fp8_noclamp_end:
.size test_cvt_sr_fp8_noclamp, .Ltest_cvt_sr_fp8_noclamp_end-test_cvt_sr_fp8_noclamp

.rodata
.p2align 8
.amdhsa_kernel test_cvt_sr_fp8_byte0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_byte2
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_noclamp
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
