// COM: Test v_cvt_f32_fp8 CLAMP=1 (E5M3) UE5M3→F32 unpack conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_f32_fp8
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: byte extraction, NaN detection, exp-31
// COM: detection, exp-31 direct F32 construction, F16 base path, exp-31
// COM: select, and NaN override.

// COM: -----------------------------------------------------------------------
// COM: Build a minimal gfx1250 code object with the target instruction.
// COM: -----------------------------------------------------------------------

// COM: Write the assembly source using printf.
// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n' > %t.s
// RUN: printf '.text\n' >> %t.s
// RUN: printf '.globl test_cvt_f32_fp8_byte0\n.p2align 8\n.type test_cvt_f32_fp8_byte0,@function\n' >> %t.s
// RUN: printf 'test_cvt_f32_fp8_byte0:\n' >> %t.s
// RUN: printf '  v_cvt_f32_fp8 v0, v1 clamp\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_f32_fp8_byte0_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_f32_fp8_byte0, .Ltest_cvt_f32_fp8_byte0_end - test_cvt_f32_fp8_byte0\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.globl test_cvt_f32_fp8_byte2\n.p2align 8\n.type test_cvt_f32_fp8_byte2,@function\n' >> %t.s
// RUN: printf 'test_cvt_f32_fp8_byte2:\n' >> %t.s
// COM: Manually encode v_cvt_f32_fp8 v5, v6 clamp with byte_sel=2.
// COM: The assembler may not set OPSEL[1:0] for byte_sel on this instruction,
// COM: so we emit the raw dwords directly:
// COM:   dword0 = 0xD5EC8805 (CLAMP=1, OPSEL[1]=0, OPSEL[0]=1 → byte_sel=2, vdst=v5)
// COM:   dword1 = 0x02010106 (src0=v6=0x106, no modifiers)
// RUN: printf '  .long 0xD5EC8805\n  .long 0x02010106\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_f32_fp8_byte2_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_f32_fp8_byte2, .Ltest_cvt_f32_fp8_byte2_end - test_cvt_f32_fp8_byte2\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.globl test_cvt_f32_fp8_noclamp\n.p2align 8\n.type test_cvt_f32_fp8_noclamp,@function\n' >> %t.s
// RUN: printf 'test_cvt_f32_fp8_noclamp:\n' >> %t.s
// RUN: printf '  v_cvt_f32_fp8 v10, v11\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_f32_fp8_noclamp_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_f32_fp8_noclamp, .Ltest_cvt_f32_fp8_noclamp_end - test_cvt_f32_fp8_noclamp\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.rodata\n.p2align 6\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_f32_fp8_byte0\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 2\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
// RUN: printf '  .amdhsa_float_round_mode_32 0\n  .amdhsa_float_round_mode_16_64 0\n' >> %t.s
// RUN: printf '  .amdhsa_float_denorm_mode_32 3\n  .amdhsa_float_denorm_mode_16_64 3\n' >> %t.s
// RUN: printf '.end_amdhsa_kernel\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_f32_fp8_byte2\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 7\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
// RUN: printf '  .amdhsa_float_round_mode_32 0\n  .amdhsa_float_round_mode_16_64 0\n' >> %t.s
// RUN: printf '  .amdhsa_float_denorm_mode_32 3\n  .amdhsa_float_denorm_mode_16_64 3\n' >> %t.s
// RUN: printf '.end_amdhsa_kernel\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_f32_fp8_noclamp\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 12\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
// RUN: printf '  .amdhsa_float_round_mode_32 0\n  .amdhsa_float_round_mode_16_64 0\n' >> %t.s
// RUN: printf '  .amdhsa_float_denorm_mode_32 3\n  .amdhsa_float_denorm_mode_16_64 3\n' >> %t.s
// RUN: printf '.end_amdhsa_kernel\n' >> %t.s

// COM: Assemble and link into a code object.  The linker script ensures
// COM: .text is the last SHF_ALLOC section so growWithTrampolines can
// COM: append trampoline code after it.
// RUN: printf 'PHDRS { ro PT_LOAD FLAGS(4); rx PT_LOAD FLAGS(5); dyn PT_DYNAMIC; }\n' > %t.lds
// RUN: printf 'SECTIONS {\n' >> %t.lds
// RUN: printf '  .dynsym : { *(.dynsym) } :ro\n' >> %t.lds
// RUN: printf '  .gnu.hash : { *(.gnu.hash) } :ro\n' >> %t.lds
// RUN: printf '  .hash : { *(.hash) } :ro\n' >> %t.lds
// RUN: printf '  .dynstr : { *(.dynstr) } :ro\n' >> %t.lds
// RUN: printf '  .rodata : { *(.rodata) } :ro\n' >> %t.lds
// RUN: printf '  .dynamic : { *(.dynamic) } :ro :dyn\n' >> %t.lds
// RUN: printf '  .text : ALIGN(256) { *(.text) } :rx\n' >> %t.lds
// RUN: printf '}\n' >> %t.lds
// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -c -x assembler %t.s -o %t.o
// RUN: ld.lld -shared -T %t.lds %t.o -o %t.co

// COM: -----------------------------------------------------------------------
// COM: Run the hotswap rewrite and dump the output.
// COM: -----------------------------------------------------------------------
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1 && \
// RUN:   hotswap-rewrite %t.co \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump-output %t.patched.co \
// RUN:   2>%t.log | %FileCheck --check-prefix=STATUS %s
// STATUS: RESULT: SUCCESS

// COM: -----------------------------------------------------------------------
// COM: Verify: CLAMP=1 byte_sel=0 patch — original site has s_branch, and
// COM: the trampoline contains the full unpack conversion sequence:
// COM:   byte extraction → NaN detect → exp-31 detect → exp-31 direct F32
// COM:   → F16 base path → exp-31 select → NaN override
// COM:
// COM: Conversion sequence (15 instructions):
// COM:   v_and_b32        (byte extraction, byte_sel=0)
// COM:   v_cmp_eq_u32     (NaN compare: byte == 0xFF)
// COM:   s_mov_b32        (save NaN flag)
// COM:   v_cmp_lt_u32     (exp-31 compare: byte > 0xF7)
// COM:   s_mov_b32        (save exp-31 flag)
// COM:   v_and_b32        (mantissa = byte & 7)
// COM:   v_lshlrev_b32    (mantissa << 20)
// COM:   v_or_b32         (0x47800000 | mantissa<<20)
// COM:   v_lshlrev_b32    (byte << 7 → F16 bits)
// COM:   v_cvt_f32_f16    (F16 → F32)
// COM:   s_mov_b32        (restore exp-31 flag)
// COM:   v_cndmask_b32    (select exp-31 vs F16 result)
// COM:   s_mov_b32        (restore NaN flag)
// COM:   v_mov_b32        (load qNaN 0x7FA3D000)
// COM:   v_cndmask_b32    (NaN override → vdst)
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=BYTE0 %s

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
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=BYTE2 %s

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
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=NOCLAMP %s

// NOCLAMP-LABEL: <test_cvt_f32_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_f32_fp8

// COM: -----------------------------------------------------------------------
// COM: Idempotency: running the rewrite a second time should succeed and
// COM: produce the same output (original sites already contain s_branch).
// COM: -----------------------------------------------------------------------
// RUN: hotswap-rewrite %t.patched.co \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump-output %t.patched2.co \
// RUN:   | %FileCheck --check-prefix=IDEMP %s
// IDEMP: RESULT: SUCCESS

// RUN: diff %t.patched.co %t.patched2.co
