// COM: HSV-009 regression for WMMA split: gfx1250 A0 must not execute
// COM: s_add_pc_i64. Two adjacent K=128 WMMAs are more than 128 KB from the
// COM: appended pool. Their two 8-byte sites merge into one 16-byte set-PC
// COM: source window, and the split pool returns through another set-PC edge.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: safe far return: reusing site-dead vcc
// LOG: set-PC forward windows: expanded 1 far site(s), merged 1 adjacent trampoline site(s), synthesized zero s_add_pc_i64
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOADD %s

// COM: Both original WMMAs are covered by one exact-size forward sequence.
// COM: The unmodified device-function return remains the pool's return path.
// DISASM-LABEL: <test_wsplit_far_adjacent>:
// DISASM-NEXT: s_mov_b64 vcc, 0
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-NEXT: v_cmp_eq_u32_e64 vcc_lo, v0, v1
// DISASM-NEXT: v_cndmask_b32_e32 v48, v0, v1, vcc_lo
// DISASM-NEXT: s_cmp_lg_u64 s[40:41], 0
// DISASM: s_set_pc_i64 s[30:31]

// COM: Each K=128 source becomes two K=64 halves in the merged pool.
// DISASM:      v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], v[32:39]
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[40:47], v[0:7], v[16:23], v[40:47]
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[40:47], v[8:15], v[24:31], v[40:47]
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64

// COM: Whole-object invariant, covering both source and pool disassembly.
// NOADD: file format elf64-amdgpu
// NOADD-NOT: s_add_pc_i64

// COM: The K=128 forms are gone, so a second rewrite is byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: A true 64-bit VCC use after the source sites keeps VCC_HI live. The
// COM: wave32 implicit-VCC normalization must not make that pair available.
// RUN: sed 's/v_cndmask_b32_e32 v48, v0, v1/s_cmp_lg_u64 vcc, 0/' %s \
// RUN:   > %t.vcc64.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.vcc64.s -o %t.vcc64.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.vcc64.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=VCC64 %s
// VCC64: safe far return unavailable
// VCC64: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wsplit_far_adjacent
.p2align 8
.type test_wsplit_far_adjacent,@function
test_wsplit_far_adjacent:
  // VCC is already part of the function's allocation and is dead across both
  // required sites. Every eligible numbered caller-clobbered pair is read on
  // the return path, forcing the ABI-safe VCC fallback.
  s_mov_b64 vcc, 0
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  v_wmma_f32_16x16x128_fp8_fp8 v[40:47], v[0:15], v[16:31], v[40:47]
  // The compare defines VCC_LO and the e32 cndmask consumes only that wave32
  // predicate. VCC_HI is dead here and may be reused with VCC_LO for a far
  // return; the MC descriptor's implicit composite VCC must not keep it live.
  v_cmp_eq_u32_e64 vcc_lo, v0, v1
  v_cndmask_b32_e32 v48, v0, v1
  s_cmp_lg_u64 s[40:41], 0
  s_cmp_lg_u64 s[42:43], 0
  s_cmp_lg_u64 s[44:45], 0
  s_cmp_lg_u64 s[46:47], 0
  s_cmp_lg_u64 s[56:57], 0
  s_cmp_lg_u64 s[58:59], 0
  s_cmp_lg_u64 s[60:61], 0
  s_cmp_lg_u64 s[62:63], 0
  s_cmp_lg_u64 s[72:73], 0
  s_cmp_lg_u64 s[74:75], 0
  s_cmp_lg_u64 s[76:77], 0
  s_cmp_lg_u64 s[78:79], 0
  s_cmp_lg_u64 s[88:89], 0
  s_cmp_lg_u64 s[90:91], 0
  s_cmp_lg_u64 s[92:93], 0
  s_cmp_lg_u64 s[94:95], 0
  s_setpc_b64 s[30:31]
  // Non-NOP filler keeps the appended pool outside s_branch reach without
  // creating a local code cave.
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_wsplit_far_adjacent_end:
.size test_wsplit_far_adjacent, .Ltest_wsplit_far_adjacent_end-test_wsplit_far_adjacent
