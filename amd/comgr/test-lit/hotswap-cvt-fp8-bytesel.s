// COM: Test all byte_sel values (0–3) for v_cvt_sr_fp8_f32 and v_cvt_f32_fp8
// COM: with CLAMP=1 (E5M3 mode).
// COM:
// COM: Verifies that each byte_sel variant produces the correct byte-merge
// COM: sequence (SR pack) or byte-extraction sequence (F32 unpack):
// COM:   SR byte_sel=0 → single v_bfi_b32 merge
// COM:   SR byte_sel=1,2,3 → v_lshlrev_b32 + v_bfi_b32 merge
// COM:   Unpack byte_sel=0 → v_and_b32 (mask 0xFF)
// COM:   Unpack byte_sel=1 → v_bfe_u32 (offset=8, width=8)
// COM:   Unpack byte_sel=2 → v_bfe_u32 (offset=16, width=8)
// COM:   Unpack byte_sel=3 → v_lshrrev_b32 (shift=24)
// COM:
// COM: See also: hotswap-cvt-sr-fp8.s, hotswap-cvt-f32-fp8.s

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// COM: =======================================================================
// COM: v_cvt_sr_fp8_f32 — stochastic-round pack, all byte_sel values
// COM: =======================================================================

// COM: -----------------------------------------------------------------------
// COM: SR byte_sel=0: single v_bfi_b32 merge (mask 0xFF).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR0 %s

// SR0-LABEL: <test_cvt_sr_fp8_byte0>:
// SR0:       s_branch
// COM: --- VCC save ---
// SR0:       s_mov_b32
// SR0-NEXT:  v_and_b32{{.*}}0x7fffffff, v1
// COM: --- Byte merge (byte_sel=0: single bfi) ---
// SR0:       v_cndmask_b32
// SR0-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore ---
// SR0-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: SR byte_sel=1: v_lshlrev_b32 + v_bfi_b32 merge (mask 0xFF00).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR1 %s

// SR1-LABEL: <test_cvt_sr_fp8_byte1>:
// SR1:       s_branch
// COM: --- VCC save + NaN detection (anchor on unique src v4) ---
// SR1:       v_and_b32{{.*}}0x7fffffff, v4
// COM: --- Byte merge (byte_sel=1: shift + bfi) ---
// SR1:       v_cndmask_b32
// SR1-NEXT:  v_lshlrev_b32
// SR1-NEXT:  v_bfi_b32 v3,
// COM: --- VCC restore ---
// SR1-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: SR byte_sel=2: v_lshlrev_b32 + v_bfi_b32 merge (mask 0xFF0000).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR2 %s

// SR2-LABEL: <test_cvt_sr_fp8_byte2>:
// SR2:       s_branch
// COM: --- VCC save + NaN detection (anchor on unique src v7) ---
// SR2:       v_and_b32{{.*}}0x7fffffff, v7
// COM: --- Byte merge (byte_sel=2: shift + bfi) ---
// SR2:       v_cndmask_b32
// SR2-NEXT:  v_lshlrev_b32
// SR2-NEXT:  v_bfi_b32 v6,
// COM: --- VCC restore ---
// SR2-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: SR byte_sel=3: v_lshlrev_b32 + v_bfi_b32 merge (mask 0xFF000000).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR3 %s

// SR3-LABEL: <test_cvt_sr_fp8_byte3>:
// SR3:       s_branch
// COM: --- VCC save + NaN detection (anchor on unique src v10) ---
// SR3:       v_and_b32{{.*}}0x7fffffff, v10
// COM: --- Byte merge (byte_sel=3: shift + bfi) ---
// SR3:       v_cndmask_b32
// SR3-NEXT:  v_lshlrev_b32
// SR3-NEXT:  v_bfi_b32 v9,
// COM: --- VCC restore ---
// SR3-NEXT:  s_mov_b32

// COM: =======================================================================
// COM: v_cvt_f32_fp8 — UE5M3->F32 unpack, all byte_sel values
// COM: =======================================================================

// COM: -----------------------------------------------------------------------
// COM: Unpack byte_sel=0: v_and_b32 extraction (mask 0xFF).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=CVT0 %s

// CVT0-LABEL: <test_cvt_f32_fp8_byte0>:
// CVT0:       s_branch
// COM: --- VCC save + Byte extraction (byte_sel=0: anchor on unique src v1) ---
// CVT0:       v_and_b32{{.*}}0xff, v1
// COM: --- VCC restore ---
// CVT0:       v_mov_b32{{.*}}0x7fa3d000
// CVT0-NEXT:  v_cndmask_b32{{.*}}v0,
// CVT0-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: Unpack byte_sel=1: v_bfe_u32 extraction (offset=8, width=8).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=CVT1 %s

// CVT1-LABEL: <test_cvt_f32_fp8_byte1>:
// CVT1:       s_branch
// COM: --- VCC save + Byte extraction (anchor on unique src v3) ---
// CVT1:       v_bfe_u32{{.*}}v3, 8, 8
// COM: --- VCC restore ---
// CVT1:       v_mov_b32{{.*}}0x7fa3d000
// CVT1-NEXT:  v_cndmask_b32{{.*}}v2,
// CVT1-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: Unpack byte_sel=2: v_bfe_u32 extraction (offset=16, width=8).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=CVT2 %s

// CVT2-LABEL: <test_cvt_f32_fp8_byte2>:
// CVT2:       s_branch
// COM: --- VCC save + Byte extraction (anchor on unique src v5) ---
// CVT2:       v_bfe_u32{{.*}}v5, 16, 8
// COM: --- VCC restore ---
// CVT2:       v_mov_b32{{.*}}0x7fa3d000
// CVT2-NEXT:  v_cndmask_b32{{.*}}v4,
// CVT2-NEXT:  s_mov_b32

// COM: -----------------------------------------------------------------------
// COM: Unpack byte_sel=3: v_lshrrev_b32 extraction (shift=24).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=CVT3 %s

// CVT3-LABEL: <test_cvt_f32_fp8_byte3>:
// CVT3:       s_branch
// COM: --- VCC save + Byte extraction (anchor on unique src v7) ---
// CVT3:       v_lshrrev_b32{{.*}}, 24, v7
// COM: --- VCC restore ---
// CVT3:       v_mov_b32{{.*}}0x7fa3d000
// CVT3-NEXT:  v_cndmask_b32{{.*}}v6,
// CVT3-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// === v_cvt_sr_fp8_f32: stochastic-round pack (byte_sel=0,1,2,3) ===

// --- SR byte_sel=0: CLAMP=1 (asm directly) ---
.globl test_cvt_sr_fp8_byte0
.p2align 8
.type test_cvt_sr_fp8_byte0,@function
test_cvt_sr_fp8_byte0:
  v_cvt_sr_fp8_f32 v0, v1, v2 clamp
  s_endpgm
.Ltest_cvt_sr_fp8_byte0_end:
.size test_cvt_sr_fp8_byte0, .Ltest_cvt_sr_fp8_byte0_end-test_cvt_sr_fp8_byte0

// --- SR byte_sel=1: CLAMP=1 (raw encoding for OPSEL[3:2]=01) ---
.globl test_cvt_sr_fp8_byte1
.p2align 8
.type test_cvt_sr_fp8_byte1,@function
test_cvt_sr_fp8_byte1:
  // v_cvt_sr_fp8_f32 v3, v4, v5 clamp byte_sel=1
  // dword0 = 0xD76BA003 (CLAMP=1, OPSEL[3]=0, OPSEL[2]=1 -> byte_sel=1, vdst=v3)
  // dword1 = 0x02020B04 (src0=v4, src1=v5, no modifiers)
  .long 0xD76BA003
  .long 0x02020B04
  s_endpgm
.Ltest_cvt_sr_fp8_byte1_end:
.size test_cvt_sr_fp8_byte1, .Ltest_cvt_sr_fp8_byte1_end-test_cvt_sr_fp8_byte1

// --- SR byte_sel=2: CLAMP=1 (raw encoding for OPSEL[3:2]=10) ---
.globl test_cvt_sr_fp8_byte2
.p2align 8
.type test_cvt_sr_fp8_byte2,@function
test_cvt_sr_fp8_byte2:
  // v_cvt_sr_fp8_f32 v6, v7, v8 clamp byte_sel=2
  // dword0 = 0xD76BC006 (CLAMP=1, OPSEL[3]=1, OPSEL[2]=0 -> byte_sel=2, vdst=v6)
  // dword1 = 0x02021107 (src0=v7, src1=v8, no modifiers)
  .long 0xD76BC006
  .long 0x02021107
  s_endpgm
.Ltest_cvt_sr_fp8_byte2_end:
.size test_cvt_sr_fp8_byte2, .Ltest_cvt_sr_fp8_byte2_end-test_cvt_sr_fp8_byte2

// --- SR byte_sel=3: CLAMP=1 (raw encoding for OPSEL[3:2]=11) ---
.globl test_cvt_sr_fp8_byte3
.p2align 8
.type test_cvt_sr_fp8_byte3,@function
test_cvt_sr_fp8_byte3:
  // v_cvt_sr_fp8_f32 v9, v10, v11 clamp byte_sel=3
  // dword0 = 0xD76BE009 (CLAMP=1, OPSEL[3]=1, OPSEL[2]=1 -> byte_sel=3, vdst=v9)
  // dword1 = 0x0202170A (src0=v10, src1=v11, no modifiers)
  .long 0xD76BE009
  .long 0x0202170A
  s_endpgm
.Ltest_cvt_sr_fp8_byte3_end:
.size test_cvt_sr_fp8_byte3, .Ltest_cvt_sr_fp8_byte3_end-test_cvt_sr_fp8_byte3

// === v_cvt_f32_fp8: UE5M3->F32 unpack (byte_sel=0,1,2,3) ===

// --- Unpack byte_sel=0: CLAMP=1 (asm directly) ---
.globl test_cvt_f32_fp8_byte0
.p2align 8
.type test_cvt_f32_fp8_byte0,@function
test_cvt_f32_fp8_byte0:
  v_cvt_f32_fp8 v0, v1 clamp
  s_endpgm
.Ltest_cvt_f32_fp8_byte0_end:
.size test_cvt_f32_fp8_byte0, .Ltest_cvt_f32_fp8_byte0_end-test_cvt_f32_fp8_byte0

// --- Unpack byte_sel=1: CLAMP=1 (raw encoding for OPSEL[1:0]=10) ---
.globl test_cvt_f32_fp8_byte1
.p2align 8
.type test_cvt_f32_fp8_byte1,@function
test_cvt_f32_fp8_byte1:
  // v_cvt_f32_fp8 v2, v3 clamp byte_sel=1
  // dword0 = 0xD5EC9002 (CLAMP=1, OPSEL[1]=1 -> byte_sel=1, vdst=v2)
  // dword1 = 0x02010103 (src0=v3, no modifiers)
  .long 0xD5EC9002
  .long 0x02010103
  s_endpgm
.Ltest_cvt_f32_fp8_byte1_end:
.size test_cvt_f32_fp8_byte1, .Ltest_cvt_f32_fp8_byte1_end-test_cvt_f32_fp8_byte1

// --- Unpack byte_sel=2: CLAMP=1 (raw encoding for OPSEL[1:0]=01) ---
.globl test_cvt_f32_fp8_byte2
.p2align 8
.type test_cvt_f32_fp8_byte2,@function
test_cvt_f32_fp8_byte2:
  // v_cvt_f32_fp8 v4, v5 clamp byte_sel=2
  // dword0 = 0xD5EC8804 (CLAMP=1, OPSEL[0]=1 -> byte_sel=2, vdst=v4)
  // dword1 = 0x02010105 (src0=v5, no modifiers)
  .long 0xD5EC8804
  .long 0x02010105
  s_endpgm
.Ltest_cvt_f32_fp8_byte2_end:
.size test_cvt_f32_fp8_byte2, .Ltest_cvt_f32_fp8_byte2_end-test_cvt_f32_fp8_byte2

// --- Unpack byte_sel=3: CLAMP=1 (raw encoding for OPSEL[1:0]=11) ---
.globl test_cvt_f32_fp8_byte3
.p2align 8
.type test_cvt_f32_fp8_byte3,@function
test_cvt_f32_fp8_byte3:
  // v_cvt_f32_fp8 v6, v7 clamp byte_sel=3
  // dword0 = 0xD5EC9806 (CLAMP=1, OPSEL[1]=1, OPSEL[0]=1 -> byte_sel=3, vdst=v6)
  // dword1 = 0x02010107 (src0=v7, no modifiers)
  .long 0xD5EC9806
  .long 0x02010107
  s_endpgm
.Ltest_cvt_f32_fp8_byte3_end:
.size test_cvt_f32_fp8_byte3, .Ltest_cvt_f32_fp8_byte3_end-test_cvt_f32_fp8_byte3

.rodata
.p2align 8
.amdhsa_kernel test_cvt_sr_fp8_byte0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_byte1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_byte2
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_byte3
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte0
  .amdhsa_next_free_vgpr 2
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte2
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte3
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
