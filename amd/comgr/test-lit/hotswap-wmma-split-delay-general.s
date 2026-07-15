// Exercise combined-delay repair from encoded dependency geometry rather than
// one compiler schedule. The positive cases vary dependency classes, skip
// spans, and intervening TRANS history. The negative cases require fail-closed
// behavior when the adjusted TRANS ordinal is not encodable or a moved member
// affects control flow.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.classes.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.classes.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.classes.out \
// RUN:   2>&1 | %FileCheck --check-prefix=CLASSES-API %s
// RUN: %llvm-objdump -d %t.classes.out \
// RUN:   | %FileCheck --check-prefix=CLASSES-DISASM %s
// CLASSES-API: WMMA split: patched {{.*}} by demerging combined delay
// CLASSES-API: RESULT: SUCCESS
// CLASSES-DISASM-LABEL: <test_wmma_delay_general>:
// CLASSES-DISASM: s_delay_alu instid0(VALU_DEP_3)
// CLASSES-DISASM-NEXT: v_mov_b32_e32 v60, v60
// CLASSES-DISASM: s_delay_alu instid0(SALU_CYCLE_2)
// CLASSES-DISASM-NEXT: s_mov_b32 s19, s19
// CLASSES-DISASM-NEXT: s_nop 0
// CLASSES-DISASM: v_wmma_f32_16x16x64_fp8_fp8
// CLASSES-DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// CLASSES-DISASM-NEXT: s_barrier_wait 0xffff

// With no later TRANS, old TRANS_DEP_2 names a producer older than the source
// WMMA. One additional split TRANS shifts it to the last encodable ordinal.
// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trans-remap.elf
// RUN: hotswap-rewrite %t.trans-remap.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode \
// RUN:   --output %t.trans-remap.out | %FileCheck --check-prefix=TRANS2-API %s
// RUN: %llvm-objdump -d %t.trans-remap.out \
// RUN:   | %FileCheck --check-prefix=TRANS2-DISASM %s
// TRANS2-API: RESULT: SUCCESS
// TRANS2-DISASM-LABEL: <test_wmma_delay_general>:
// TRANS2-DISASM: s_delay_alu instid0(TRANS32_DEP_3)
// TRANS2-DISASM-NEXT: v_mov_b32_e32 v60, v60

// A later TRANS makes the original WMMA the second prior TRANS; its semantic
// producer maps to the last split half and therefore keeps DEP_2 unchanged.
// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trans-later.elf
// RUN: hotswap-rewrite %t.trans-later.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode \
// RUN:   --output %t.trans-later.out | %FileCheck --check-prefix=LATER-API %s
// RUN: %llvm-objdump -d %t.trans-later.out \
// RUN:   | %FileCheck --check-prefix=LATER-DISASM %s
// LATER-API: RESULT: SUCCESS
// LATER-DISASM-LABEL: <test_wmma_delay_general>:
// LATER-DISASM: s_delay_alu instid0(TRANS32_DEP_2)
// LATER-DISASM-NEXT: v_mov_b32_e32 v60, v60
// LATER-DISASM: v_wmma_f32_16x16x64_fp8_fp8
// LATER-DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// LATER-DISASM-NEXT: v_sin_f32_e32 v50, v51

// TRANS_DEP_3 with no later TRANS would require unencodable DEP_4.
// RUN: %clang -x assembler-with-cpp -DCASE=4 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trans-overflow.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.trans-overflow.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=TRANS3 %s
// TRANS3: has an unrepresentable TRANS dependency after split
// TRANS3: RESULT: ERROR

// A branch inside the protected interval cannot be moved into the trampoline.
// RUN: %clang -x assembler-with-cpp -DCASE=5 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.control.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.control.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=CONTROL %s
// CONTROL: has a non-relocatable instruction in its delay window
// CONTROL: RESULT: ERROR

// The second dependency may be reconstructed in a generic dword slot. When
// that slot contains the canonical scalar delay, tensor masking may consume it
// even though the immutable instruction there was the moved predecessor.
// RUN: %clang -x assembler-with-cpp -DCASE=6 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.tensor.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite \
// RUN:   %t.tensor.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.tensor.out \
// RUN:   2>&1 | %FileCheck --check-prefix=TENSOR-API %s
// RUN: %llvm-objdump -d %t.tensor.out \
// RUN:   | %FileCheck --check-prefix=TENSOR-DISASM %s
// TENSOR-API: WMMA split: patched {{.*}} by demerging combined delay
// TENSOR-API: tensor_load_to_lds: in-place descriptor mask
// TENSOR-API: RESULT: SUCCESS
// TENSOR-DISASM-LABEL: <test_wmma_delay_general>:
// TENSOR-DISASM: s_pack_hh_b32_b16 s4, 0, s4
// TENSOR-DISASM-NEXT: tensor_load_to_lds s[24:27], s[4:11]

// A non-callable symbol can own bytes even when its start lies before the
// source head. Its extent, not only its address, blocks relocation.
// RUN: %clang -x assembler-with-cpp -DCASE=7 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.extent.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.extent.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=EXTENT %s
// EXTENT: overlaps a sized non-callable text symbol
// EXTENT: RESULT: ERROR

// If the WMMA is itself the second target, both reconstructed dependencies
// move before the replacement and no TRANS ordinal adjustment is needed.
// RUN: %clang -x assembler-with-cpp -DCASE=8 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.second-target.elf
// RUN: hotswap-rewrite %t.second-target.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode \
// RUN:   --output %t.second-target.out \
// RUN:   | %FileCheck --check-prefix=SECOND-API %s
// RUN: %llvm-objdump -d %t.second-target.out \
// RUN:   | %FileCheck --check-prefix=SECOND-DISASM %s
// SECOND-API: RESULT: SUCCESS
// SECOND-DISASM-LABEL: <test_wmma_delay_general>:
// SECOND-DISASM: s_delay_alu instid0(VALU_DEP_1)
// SECOND-DISASM-NEXT: s_mov_b32 s19, s19
// SECOND-DISASM-NEXT: s_nop 0
// SECOND-DISASM-NEXT: s_delay_alu instid0(TRANS32_DEP_2)
// SECOND-DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// SECOND-DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8

// gfx1250 non-XDL WMMA uses the VALU pipeline and must not advance a TRANS
// dependency ordinal. With no later TRANS, DEP_2 still shifts to DEP_3.
// RUN: %clang -x assembler-with-cpp -DCASE=9 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.non-xdl.elf
// RUN: hotswap-rewrite %t.non-xdl.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode \
// RUN:   --output %t.non-xdl.out | %FileCheck --check-prefix=NON-XDL-API %s
// RUN: %llvm-objdump -d %t.non-xdl.out \
// RUN:   | %FileCheck --check-prefix=NON-XDL-DISASM %s
// NON-XDL-API: RESULT: SUCCESS
// NON-XDL-DISASM-LABEL: <test_wmma_delay_general>:
// NON-XDL-DISASM: s_delay_alu instid0(TRANS32_DEP_3)
// NON-XDL-DISASM-NEXT: s_barrier_wait 0xffff

// F64 DPMACC may carry the TRANS TSFlag but AMDGPUInsertDelayAlu classifies it
// as ordinary VALU. It likewise must not hide the required DEP_2 -> DEP_3
// adjustment.
// RUN: %clang -x assembler-with-cpp -DCASE=10 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.dpmacc.elf
// RUN: hotswap-rewrite %t.dpmacc.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 --strict-mode \
// RUN:   --output %t.dpmacc.out | %FileCheck --check-prefix=DPMACC-API %s
// RUN: %llvm-objdump -d %t.dpmacc.out \
// RUN:   | %FileCheck --check-prefix=DPMACC-DISASM %s
// DPMACC-API: RESULT: SUCCESS
// DPMACC-DISASM-LABEL: <test_wmma_delay_general>:
// DPMACC-DISASM: s_delay_alu instid0(TRANS32_DEP_3)
// DPMACC-DISASM-NEXT: s_barrier_wait 0xffff

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_delay_general
.p2align 8
.type test_wmma_delay_general,@function
test_wmma_delay_general:
#if CASE == 1
  s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
  s_mov_b32 s19, s19
  s_nop 0
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  v_mov_b32 v60, v60
#elif CASE == 2
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  v_mov_b32 v60, v60
#elif CASE == 3
  s_delay_alu instid0(FMA_ACCUM_CYCLE_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  v_sin_f32 v50, v51
  s_barrier_wait 0xffff
  v_mov_b32 v60, v60
#elif CASE == 4
  s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(TRANS32_DEP_3)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  v_mov_b32 v60, v60
#elif CASE == 5
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_branch .Ltarget
  s_barrier_wait 0xffff
.Ltarget:
  v_mov_b32 v60, v60
#elif CASE == 6
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
#elif CASE == 7
  .globl protected_text_object
  .type protected_text_object,@object
protected_text_object:
  s_nop 0
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  v_mov_b32 v60, v60
.Lprotected_text_object_end:
  .size protected_text_object, .Lprotected_text_object_end-protected_text_object
#elif CASE == 8
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
  s_mov_b32 s19, s19
  s_nop 0
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
#elif CASE == 9
  s_delay_alu instid0(FMA_ACCUM_CYCLE_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  v_wmma_f32_16x16x4_f32 v[40:47], v[48:49], v[50:51], v[40:47]
  s_nop 0
  s_barrier_wait 0xffff
#elif CASE == 10
  s_delay_alu instid0(FMA_ACCUM_CYCLE_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_2)
  s_mov_b32 s19, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  v_rcp_f64 v[48:49], v[50:51]
  s_nop 0
  s_barrier_wait 0xffff
#else
  .error "CASE must select a test body"
#endif
  s_endpgm
.Ltest_wmma_delay_general_end:
.size test_wmma_delay_general, .Ltest_wmma_delay_general_end-test_wmma_delay_general
