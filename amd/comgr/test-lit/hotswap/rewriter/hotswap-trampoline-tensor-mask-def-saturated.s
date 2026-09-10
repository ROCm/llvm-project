// COM: The motivating case: the SGPR budget is saturated to s106 and the
// COM: descriptor feeds two tensors, so the at-site fallback cannot safely
// COM: modify the first use without a scratch register. The definition-time
// COM: clear applies because both consumers require the cleared descriptor.
// COM: There is no NOP sled or spare SGPR.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

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
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]

// COM: Idempotency: a second rewrite of the shared saturated descriptor makes
// COM: no further change.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

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
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
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
