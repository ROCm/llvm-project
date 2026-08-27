// COM: A0 tensor multicast-mask clearing with the descriptor in an alternate
// COM: SReg_256 range (s[16:23]). Verifies getDescriptorBaseSgpr extracts s16
// COM: and the mask-set for that base is cleared, not a fixed s4.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s16)
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_mask_alt_base>:
// DISASM: s_and_b32 s16, s16, 0xfff70000
// DISASM: tensor_load_to_lds s[0:3], s[16:23]

// COM: Idempotency: a second rewrite of the alternate-base clear makes no
// COM: further change.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_alt_base
.p2align 8
.type test_tensor_mask_alt_base,@function
test_tensor_mask_alt_base:
  s_mov_b32 s16, 0
  s_and_b32 s16, s16, 0xfffcffff
  s_or_b32 s16, s16, s5
  s_and_b32 s16, s16, 0xfff7ffff
  tensor_load_to_lds s[0:3], s[16:23]
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_mask_alt_base_end:
.size test_tensor_mask_alt_base, .Ltest_tensor_mask_alt_base_end-test_tensor_mask_alt_base

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_alt_base
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_alt_base
      .symbol: test_tensor_mask_alt_base.kd
      .sgpr_count: 24
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
