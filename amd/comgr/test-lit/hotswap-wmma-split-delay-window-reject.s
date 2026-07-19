// A protected WMMA is safe to split only when its immediately preceding
// single-instruction delay can move with it as one unambiguous source window.
// Assemble ten structural counterexamples from this file and require each to
// fail closed.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.direct.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.direct.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=DIRECT %s
// DIRECT: WMMA split: protected site at 0x{{[0-9A-F]+}} has a direct entry into the source-window interior
// DIRECT: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.overlap.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.overlap.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=OVERLAP %s
// OVERLAP: WMMA split: protected site at 0x{{[0-9A-F]+}} overlaps another delay or hard clause
// OVERLAP: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.span.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.span.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=SPAN %s
// SPAN: WMMA split: protected site at 0x{{[0-9A-F]+}} has no supported preceding delay window
// SPAN: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=4 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.indirect.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.indirect.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=INDIRECT %s
// INDIRECT: WMMA split: protected site at 0x{{[0-9A-F]+}} cannot be relocated with an unresolved indirect entry
// INDIRECT: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=5 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.combined-entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.combined-entry.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=COMBINED-ENTRY %s
// COMBINED-ENTRY: WMMA split: protected site at 0x{{[0-9A-F]+}} has a direct entry into the source-window interior
// COMBINED-ENTRY: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=6 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.combined-def.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.combined-def.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=COMBINED-DEF %s
// COMBINED-DEF: descriptor definition at 0x{{[0-9A-F]+}} is relocated; using linked delay slot for the mask
// COMBINED-DEF: is not preceded by the canonical scalar delay
// COMBINED-DEF: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=7 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.missing-first-id.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.missing-first-id.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=BAD-ID %s

// RUN: %clang -x assembler-with-cpp -DCASE=8 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.missing-second-id.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.missing-second-id.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=BAD-ID %s
// BAD-ID: WMMA split: protected site at 0x{{[0-9A-F]+}} {{(has no supported preceding delay window|has an unsupported combined dependency graph)}}
// BAD-ID: RESULT: ERROR

// COM: A successful WMMA demerge followed by a required tensor-mask failure
// COM: must not write the partially mutated in-memory object to the output.
// RUN: %clang -x assembler-with-cpp -DCASE=9 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.late-failure.elf
// RUN: rm -f %t.late-failure.out.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.late-failure.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.late-failure.out.elf \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=LATE-FAIL %s
// RUN: test ! -e %t.late-failure.out.elf
// LATE-FAIL: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} by demerging combined delay
// LATE-FAIL: tensor_load_to_lds at 0x{{[0-9A-F]+}} is not preceded by the canonical scalar delay
// LATE-FAIL: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=10 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trans32-dependency.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.trans32-dependency.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=TRANS32 %s
// TRANS32: has an unrepresentable TRANS dependency after split
// TRANS32: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_delay_reject
.p2align 8
.type test_wmma_delay_reject,@function
test_wmma_delay_reject:
#if CASE == 1
  // Target the first literal slot, not merely the WMMA instruction boundary.
  // s_branch simm16=2 reaches PC+12 from this instruction at PC+0.
  .long 0xBFA00002
  s_delay_alu instid0(VALU_DEP_1)
.Lwmma:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
#elif CASE == 2
  s_delay_alu instid0(SALU_CYCLE_1)
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
#elif CASE == 3
  // Reserved dependency ID 12 is malformed and retains the conservative
  // maximum protection span, but cannot be reconstructed.
  s_delay_alu 0xc
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  v_nop
  v_nop
  v_nop
  v_nop
  v_nop
#elif CASE == 4
  s_set_pc_i64 s[0:1]
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
#elif CASE == 5
  // The predecessor is overwritten with the demerged tensor delay, so an
  // independent edge to that slot must reject the complete transformation.
  s_branch .Lcombined_barrier
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
.Lcombined_barrier:
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 6
  // Tensor masking would otherwise try to patch this definition separately,
  // overlapping the WMMA source window.
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_readfirstlane_b32 s4, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 7
  // Both dependency IDs are required by the demerge proof.
  s_delay_alu 0xb0
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 8
  // A skip field without instid1 is malformed and retains the conservative
  // six-instruction relocation protection.
  s_delay_alu 0x32
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 9
  // No straight-line definition of tensor base s4 exists. The WMMA demerge
  // is valid, but the later required tensor mask must fail the whole rewrite.
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 10
  // With no later TRANS, splitting one WMMA into two would shift an older
  // TRANS_DEP_3 producer to unencodable ordinal four.
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_3)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#else
  .error "CASE must select a test body"
#endif
  s_endpgm
.Ltest_wmma_delay_reject_end:
.size test_wmma_delay_reject, .Ltest_wmma_delay_reject_end-test_wmma_delay_reject
