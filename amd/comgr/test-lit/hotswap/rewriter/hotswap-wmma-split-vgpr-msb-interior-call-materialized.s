// A PC-materialized s_swap_pc_i64 that calls into the interior of another
// function skips that function's s_set_vgpr_msb prefix. The VGPR-MSB analysis
// must treat the callee as entered at a non-start offset and refuse to seed it
// from the symbol start, so a mandatory WMMA split there fails closed instead
// of bracketing for the wrong incoming mode.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
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
  // Materialize the interior label of test_callee and call it. s_get_pc_i64
  // captures the address of the following instruction (.Lcaller_pc), so the
  // add displacement is (target - captured PC).
  s_get_pc_i64 s[4:5]
.Lcaller_pc:
  s_add_nc_u64 s[4:5], s[4:5], .Lcallee_interior-.Lcaller_pc
  s_swap_pc_i64 s[0:1], s[4:5]
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
