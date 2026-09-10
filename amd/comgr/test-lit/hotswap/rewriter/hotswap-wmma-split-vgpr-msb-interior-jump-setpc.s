// A PC-materialized s_set_pc_i64 jump (not a call) into the interior of another
// function is not tracked as an unresolved call target, so it does not trip the
// object-wide unresolved-target guard. The VGPR-MSB analysis must still resolve
// the jump's materialized target over the rewrite's control-flow surface, see
// that it enters test_target at a non-start offset (skipping the s_set_vgpr_msb
// prefix), and decline the mandatory WMMA split there. Analyzing test_target
// from its symbol start would emit brackets for the wrong incoming mode.
//
// CHECK-NOT verifies the decline comes from the interior-entry resolver, not
// from the unresolved-call bail: no "unresolved call target" is logged here.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s
// RUN: test ! -e %t.out.elf
// CHECK-NOT: unresolved call target
// CHECK-LABEL: WMMA split: cannot determine VGPR-MSB mode
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_source
.p2align 8
.type test_source,@function
test_source:
  // Materialize test_target's interior (.text+0x104) and tail-jump to it. The
  // captured PC is the s_add address (.text+0x4), so the +0x100 displacement
  // lands past test_target's s_set_vgpr_msb prefix. The small inline immediate
  // keeps the compiler-canonical s_add form the target resolver understands.
  s_get_pc_i64 s[0:1]
  s_add_nc_u64 s[0:1], s[0:1], 0x100
  s_set_pc_i64 s[0:1]
.size test_source, .-test_source

.globl test_target
.p2align 8
.type test_target,@function
test_target:
  s_set_vgpr_msb 0x60
.Ltarget_interior:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_target, .-test_target
