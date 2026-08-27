// COM: Tensor CFGs may consume exact target sets from the object-wide bounded
// COM: return audit. The default helper returns to an in-function continuation,
// COM: so its otherwise-indirect set-PC is a closed edge and the zero-mask
// COM: proof succeeds. The variant's proven return target is outside the
// COM: tensor function, which this local CFG rejects conservatively.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOCAL %s
// LOCAL: descriptor workgroup_mask is already zero on every path
// LOCAL: hotswap: applied 0 instruction patches
// LOCAL: RESULT: SUCCESS
// RUN: cmp %t.elf %t.out.elf

// RUN: sed 's/^\.set outside_return, 0$/.set outside_return, 1/' \
// RUN:   %s > %t.outside.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.outside.s -o %t.outside.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.outside.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.outside.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OUTSIDE %s
// OUTSIDE: tensor CFG rejected bounded transfer at 0x{{[0-9A-F]+}} with out-of-range target
// OUTSIDE-NOT: descriptor workgroup_mask is already zero on every path
// OUTSIDE: tensor_load_to_lds: s4 dead, no save/restore needed
// OUTSIDE: RESULT: SUCCESS

.set outside_return, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.if outside_return
.type tensor_bounded_return_callee,@function
tensor_bounded_return_callee:
  s_mov_b32 s4, 0
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s4, 0
  s_set_pc_i64 s[0:1]
.Ltensor_bounded_return_callee_end:
.size tensor_bounded_return_callee, .Ltensor_bounded_return_callee_end-tensor_bounded_return_callee

.globl test_tensor_mask_bounded_return
.p2align 8
.type test_tensor_mask_bounded_return,@function
test_tensor_mask_bounded_return:
.Loutside_getpc:
  s_get_pc_i64 s[2:3]
  // The aligned caller begins at .text+0x100; captured PC is .text+0x104.
  s_add_nc_u64 s[2:3], s[2:3], -260
  s_swap_pc_i64 s[0:1], s[2:3]
  s_endpgm
.Ltest_tensor_mask_bounded_return_end:
.size test_tensor_mask_bounded_return, .Ltest_tensor_mask_bounded_return_end-test_tensor_mask_bounded_return
.else
.globl test_tensor_mask_bounded_return
.p2align 8
.type test_tensor_mask_bounded_return,@function
test_tensor_mask_bounded_return:
  s_mov_b32 s4, 0
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s4, 0
.Llocal_getpc:
  s_get_pc_i64 s[2:3]
  // Captured PC is .text+0x18 and the helper begins at .text+0x24.
  s_add_nc_u64 s[2:3], s[2:3], 12
  s_swap_pc_i64 s[0:1], s[2:3]
.Lcontinuation:
  s_endpgm
.Lhelper:
  s_set_pc_i64 s[0:1]
.Ltest_tensor_mask_bounded_return_end:
.size test_tensor_mask_bounded_return, .Ltest_tensor_mask_bounded_return_end-test_tensor_mask_bounded_return
.endif

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_bounded_return
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_bounded_return
      .symbol: test_tensor_mask_bounded_return.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
