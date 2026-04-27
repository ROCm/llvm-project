// COM: Test v_cvt_sr_fp8_f32 CLAMP=1 (E5M3) stochastic-round conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_sr_fp8_f32
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: NaN detection, stochastic noise injection,
// COM: F32→F16→UE5M3 conversion, overflow clamping, NaN override, and byte merge.

// COM: -----------------------------------------------------------------------
// COM: Build a minimal gfx1250 code object with the target instruction.
// COM: -----------------------------------------------------------------------

// COM: Write the assembly source using printf.
// RUN: printf '.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n' > %t.s
// RUN: printf '.text\n' >> %t.s
// RUN: printf '.globl test_cvt_sr_fp8_byte0\n.p2align 8\n.type test_cvt_sr_fp8_byte0,@function\n' >> %t.s
// RUN: printf 'test_cvt_sr_fp8_byte0:\n' >> %t.s
// RUN: printf '  v_cvt_sr_fp8_f32 v0, v1, v2 clamp\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_sr_fp8_byte0_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_sr_fp8_byte0, .Ltest_cvt_sr_fp8_byte0_end - test_cvt_sr_fp8_byte0\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.globl test_cvt_sr_fp8_byte2\n.p2align 8\n.type test_cvt_sr_fp8_byte2,@function\n' >> %t.s
// RUN: printf 'test_cvt_sr_fp8_byte2:\n' >> %t.s
// COM: Manually encode v_cvt_sr_fp8_f32 v5, v6, v7 clamp with byte_sel=2.
// COM: The assembler may not set OPSEL[3:2] for the non-t16 variant,
// COM: so we emit the raw dwords directly:
// COM:   dword0 = 0xD76BC005 (CLAMP=1, OPSEL[3]=1, OPSEL[2]=0 → byte_sel=2, vdst=v5)
// COM:   dword1 = 0x02020F06 (src0=v6=0x106, src1=v7=0x107, no modifiers)
// RUN: printf '  .long 0xD76BC005\n  .long 0x02020F06\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_sr_fp8_byte2_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_sr_fp8_byte2, .Ltest_cvt_sr_fp8_byte2_end - test_cvt_sr_fp8_byte2\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.globl test_cvt_sr_fp8_noclamp\n.p2align 8\n.type test_cvt_sr_fp8_noclamp,@function\n' >> %t.s
// RUN: printf 'test_cvt_sr_fp8_noclamp:\n' >> %t.s
// RUN: printf '  v_cvt_sr_fp8_f32 v10, v11, v12\n' >> %t.s
// RUN: printf '  s_endpgm\n' >> %t.s
// RUN: printf '.Ltest_cvt_sr_fp8_noclamp_end:\n' >> %t.s
// RUN: printf '  .size test_cvt_sr_fp8_noclamp, .Ltest_cvt_sr_fp8_noclamp_end - test_cvt_sr_fp8_noclamp\n' >> %t.s
// RUN: printf '\n' >> %t.s
// RUN: printf '.rodata\n.p2align 6\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_sr_fp8_byte0\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 3\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
// RUN: printf '  .amdhsa_float_round_mode_32 0\n  .amdhsa_float_round_mode_16_64 0\n' >> %t.s
// RUN: printf '  .amdhsa_float_denorm_mode_32 3\n  .amdhsa_float_denorm_mode_16_64 3\n' >> %t.s
// RUN: printf '.end_amdhsa_kernel\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_sr_fp8_byte2\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 8\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
// RUN: printf '  .amdhsa_float_round_mode_32 0\n  .amdhsa_float_round_mode_16_64 0\n' >> %t.s
// RUN: printf '  .amdhsa_float_denorm_mode_32 3\n  .amdhsa_float_denorm_mode_16_64 3\n' >> %t.s
// RUN: printf '.end_amdhsa_kernel\n' >> %t.s
// RUN: printf '.amdhsa_kernel test_cvt_sr_fp8_noclamp\n' >> %t.s
// RUN: printf '  .amdhsa_group_segment_fixed_size 0\n  .amdhsa_private_segment_fixed_size 0\n' >> %t.s
// RUN: printf '  .amdhsa_kernarg_size 0\n  .amdhsa_next_free_vgpr 13\n  .amdhsa_next_free_sgpr 2\n' >> %t.s
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
// COM: the trampoline contains the full SR conversion sequence:
// COM:   NaN detect → clamp → noise inject → F16 → UE5M3 → overflow clamp
// COM:   → NaN override → byte merge
// COM:
// COM: Conversion sequence (18 instructions):
// COM:   v_and_b32        (strip sign for NaN test)
// COM:   v_cmp_lt_u32     (NaN compare)
// COM:   s_mov_b32        (save NaN flag)
// COM:   v_max_num_f32    (clamp negative)
// COM:   v_and_b32        (extract F32 mantissa)
// COM:   v_lshrrev_b32    (shift noise seed S1[31:12])
// COM:   v_add_*_u32      (add noise to mantissa)
// COM:   v_and_b32        (truncate mantissa to 23 bits)
// COM:   v_max_num_f32    (reload clamped src0)
// COM:   v_bfi_b32        (replace mantissa, keep sign+exp)
// COM:   v_cvt_f16_f32    (F32 → F16)
// COM:   v_lshrrev_b32    (F16 → UE5M3 byte)
// COM:   v_min_u32        (overflow clamp to 0xFE)
// COM:   s_mov_b32        (restore NaN flag)
// COM:   v_mov_b32        (load NaN byte 0xFF)
// COM:   v_cndmask_b32    (NaN override)
// COM:   v_bfi_b32        (byte merge into vdst)
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=BYTE0 %s

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
// COM: --- F32 → F16 → UE5M3 ---
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
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=BYTE2 %s

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
// COM: --- F32 → F16 → UE5M3 ---
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
// RUN: %llvm-objdump -d --no-leading-addr --no-show-raw-insn %t.patched.co \
// RUN:   | %FileCheck --check-prefix=NOCLAMP %s

// NOCLAMP-LABEL: <test_cvt_sr_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_sr_fp8_f32

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
