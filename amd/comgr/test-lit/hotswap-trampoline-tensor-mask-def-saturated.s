// COM: The motivating case: the SGPR budget is saturated to s106 and the
// COM: descriptor SGPR is live after the tensor, so the at-site fallback would
// COM: fail for lack of a scratch register. Because the descriptor is built by
// COM: an in-function construction region, the definition-time clear applies
// COM: and succeeds with no scratch and no sled -- exactly what the at-site
// COM: strategy cannot do on the compute-bound kernels this fix targets.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API-NOT: no scratch SGPR available
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_mask_saturated>:
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM-NOT: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds s[0:3], s[4:11]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_saturated
.p2align 8
.type test_tensor_mask_saturated,@function
test_tensor_mask_saturated:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s8, s4
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_mask_saturated_end:
.size test_tensor_mask_saturated, .Ltest_tensor_mask_saturated_end-test_tensor_mask_saturated

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_saturated
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_saturated
      .symbol: test_tensor_mask_saturated.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
