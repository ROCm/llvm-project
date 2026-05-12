// COM: Passthrough test for the VOP3PX2 scale_src2 bit-field fix. A kernel
// COM: with no V_WMMA_SCALE* instructions must not have the in-place
// COM: scale_src2 patch fire on it.
// COM:
// COM: Since the VOP3PX2 wrap pass landed (comgr-hotswap-patch-vop3px-
// COM: wrap.cpp), the bare f8f6f4 in this kernel now gets wrapped into a
// COM: VOP3PX2 in a trampoline; the resulting v_wmma_scale_* carries
// COM: scale_src2 = VGPR0 (0x100) baked into the wrap pass's prefix bytes
// COM: (the same SALU-stall workaround the in-place vop3px2-src2 fix
// COM: applies to user-emitted VOP3PX2). So the in-place vop3px2-src2 fix
// COM: still has nothing to do here -- it only fires on user-emitted
// COM: v_wmma_scale_* instructions in the kernel body, not on
// COM: wrap-emitted ones in trampolines.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Kernel body: bare f8f6f4 becomes an s_branch into the wrap trampoline.
// COM: Trampoline body: v_wmma_scale_* (the wrap-produced fused VOP3PX2);
// COM: this is from the VOP3PX2 wrap pass, not from the in-place
// COM: scale_src2 patch.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_vop3px2_noop>:
// DISASM:       s_branch
// DISASM:       s_endpgm
// DISASM:       v_wmma_scale_f32_16x16x128_f8f6f4{{.*}}, 0, 0{{.*}}matrix_a_fmt:MATRIX_FMT_BF8{{.*}}matrix_b_fmt:MATRIX_FMT_FP6

// COM: Idempotency: second rewrite must produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_vop3px2_noop
.p2align 8
.type test_vop3px2_noop,@function
test_vop3px2_noop:
  // Regular (non-scale) WMMA: patch must not touch this.
  v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[0:7] matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6
  s_endpgm
.Ltest_vop3px2_noop_end:
.size test_vop3px2_noop, .Ltest_vop3px2_noop_end-test_vop3px2_noop

.rodata
.p2align 8
.amdhsa_kernel test_vop3px2_noop
  .amdhsa_next_free_vgpr 36
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
