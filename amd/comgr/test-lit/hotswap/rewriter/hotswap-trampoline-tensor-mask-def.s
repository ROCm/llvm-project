// COM: Test HotSwap A0 tensor_load_to_lds multicast-mask clearing at the
// COM: descriptor definition. The kernel builds the group descriptor inline
// COM: (s_mov / s_and / s_or / s_and) as TensileLite kernels do; the last
// COM: low16-preserving s_and is the mask-set. The pass clears its low 16 bits
// COM: (0xNNNNffff -> 0xNNNN0000), forcing workgroup_mask to zero, and leaves
// COM: the PC-sensitive tensor instruction untouched.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The mask-set s_and is cleared in place (0xfff7ffff -> 0xfff70000); the
// COM: pre-mask normalize keeps the low half, and the tensor instruction is
// COM: unchanged (no s_pack_hh, no save/restore, no relocation).
// DISASM-LABEL: <test_tensor_mask_def>:
// DISASM: s_and_b32 s4, s4, 0xfffcffff
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM-NOT: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds s[0:3], s[4:11]

// COM: Idempotency: a second rewrite makes no further change.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_def
.p2align 8
.type test_tensor_mask_def,@function
test_tensor_mask_def:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_mask_def_end:
.size test_tensor_mask_def, .Ltest_tensor_mask_def_end-test_tensor_mask_def

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_def
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_def
      .symbol: test_tensor_mask_def.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
