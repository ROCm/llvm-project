// s_swap_pc_i64 also accepts an absolute-immediate target. A call that lands
// in the interior of another function bypasses its s_set_vgpr_msb prefix, so
// the VGPR-MSB analysis must resolve the absolute target over the same surface
// the rewrite uses, mark the callee as entered at a non-start offset, and fail
// a mandatory WMMA split there closed rather than seed the wrong mode.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   -Wl,--section-start=.text=0x1000 %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s
// RUN: test ! -e %t.out.elf
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_caller
.p2align 8
.type test_caller,@function
test_caller:
  // test_caller sits at the .text start (0x1000), so this folds to the
  // absolute virtual address of .Lcallee_interior.
  s_swap_pc_i64 s[0:1], .Lcallee_interior-test_caller+0x1000
  s_endpgm
.size test_caller, .-test_caller

.globl test_callee
.p2align 8
.type test_callee,@function
test_callee:
  s_set_vgpr_msb 0x60
.Lcallee_interior:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_callee, .-test_callee
