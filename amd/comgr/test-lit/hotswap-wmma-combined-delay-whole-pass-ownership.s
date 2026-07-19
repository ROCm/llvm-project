// A per-instruction WMMA split can atomically relocate every instruction in a
// combined-delay window before whole-function passes run. Those later passes
// still analyze the immutable decoded stream, so they must honor ownership of
// the linked addresses rather than queue a second replacement for bytes that
// are reserved for the source branch. Exercise an overlapping co-exec VALU, an
// overlapping VOP3PX2 scale instruction, and a disjoint co-exec VALU.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.hazard.elf
// RUN: rm -f %t.hazard.out.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.hazard.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.hazard.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=HAZARD %s
// RUN: test ! -e %t.hazard.out.elf
// HAZARD: WMMA co-exec hazard at 0x{{[0-9A-F]+}}
// HAZARD: WMMA co-exec validation: 1 hazards (2 WMMA instructions scanned)
// HAZARD: WMMA split: delay-window member at 0x{{[0-9A-F]+}} requires a separate HotSwap patch
// HAZARD: WMMA split: protected site at 0x{{[0-9A-F]+}} would suppress another required HotSwap patch
// HAZARD: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.scale.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.scale.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.scale.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=SCALE %s
// SCALE: VOP3PX2 SRC2 fix at 0x{{[0-9A-F]+}}: v_wmma_scale_f32_16x16x128_f8f6f4 scale_src2 -> VGPR0
// SCALE: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} by demerging combined delay
// SCALE: RESULT: SUCCESS

// The source interval is one branch plus seven padding dwords. If the later
// bit-field pass instead touched the stale VOP3PX2 address, one of these nops
// would be corrupted. The scale instruction itself survives in the relocated
// body after being corrected before the source window is copied.
// RUN: %llvm-objdump -d %t.scale.out.elf \
// RUN:   | %FileCheck --check-prefix=SCALE-DISASM %s
// SCALE-DISASM-LABEL: <test_whole_pass_ownership>:
// SCALE-DISASM-NEXT: s_branch
// SCALE-DISASM-COUNT-7: s_nop 0
// SCALE-DISASM-NEXT: s_delay_alu instid0(VALU_DEP_1)
// SCALE-DISASM: v_wmma_scale_f32_16x16x128_f8f6f4
// SCALE-DISASM: v_wmma_f32_16x16x64_fp8_fp8
// SCALE-DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8

// RUN: hotswap-rewrite %t.scale.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.disjoint.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.disjoint.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.disjoint.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=DISJOINT %s
// DISJOINT: WMMA co-exec hazard at 0x{{[0-9A-F]+}}
// DISJOINT: WMMA co-exec validation: 1 hazards (2 WMMA instructions scanned)
// DISJOINT: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} by demerging combined delay
// DISJOINT: WMMA co-exec requirement composed into replacement at 0x{{[0-9A-F]+}} (8 leading v_nop(s))
// DISJOINT: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.disjoint.out.elf \
// RUN:   | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_whole_pass_ownership>:
// DISASM: v_wmma_i32_16x16x64_iu8
// DISASM-NEXT: s_branch
// DISASM: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-COUNT-8: v_nop
// DISASM-NEXT: v_add_f32{{(_e32)?}} v16, v0, v1

// RUN: hotswap-rewrite %t.disjoint.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_whole_pass_ownership
.p2align 8
.type test_whole_pass_ownership,@function
test_whole_pass_ownership:
#if CASE == 1
  // The integer WMMA's first overlapping VALU is also the first target of the
  // combined delay owned by the later split. The whole-function hazard pass
  // must not install another trampoline at its stale linked address.
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_add_f32 v16, v0, v1
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_nop 0
  v_mov_b32 v61, v60
#elif CASE == 2
  // The VOP3PX2 field fix is in-place and same-size. It must run before the
  // combined window is copied, so the corrected bytes move with the window
  // and no later pass mutates the linked address reserved for its source edge.
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  v_mov_b32 v63, v62
  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6 matrix_a_scale:MATRIX_SCALE_ROW1 matrix_b_scale:MATRIX_SCALE_ROW1
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_nop 0
  v_mov_b32 v61, v60
#elif CASE == 3
  // A hazard entirely after the claimed source window is independent and
  // remains eligible for its own trampoline.
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_mov_b32 v63, v62
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_nop 0
  v_mov_b32 v61, v60
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
  v_add_f32 v16, v0, v1
#else
  .error "CASE must select a test body"
#endif
  s_endpgm
.Ltest_whole_pass_ownership_end:
.size test_whole_pass_ownership, .Ltest_whole_pass_ownership_end-test_whole_pass_ownership

.rodata
.p2align 8
.amdhsa_kernel test_whole_pass_ownership
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_whole_pass_ownership
      .symbol: test_whole_pass_ownership.kd
      .sgpr_count: 2
      .vgpr_count: 64
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
