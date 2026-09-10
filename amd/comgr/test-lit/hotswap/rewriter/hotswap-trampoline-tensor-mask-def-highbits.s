// COM: The real TensileLite construction sets high descriptor bits with an
// COM: s_or base, base, imm (low 16 zero) AFTER the low16-preserving mask-set.
// COM: That trailing s_or must not disqualify the definition-time clear: it
// COM: cannot restore a nonzero workgroup_mask. The mask-set is still cleared
// COM: in place and the high-bit s_or is left intact.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_mask_highbits>:
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM-NEXT: s_or_b32 s4, s4, 0x7500000
// DISASM: tensor_load_to_lds s[0:3], s[4:11]

// COM: Idempotency: the high-bit s_or construction round-trips byte-equal on a
// COM: second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_highbits
.p2align 8
.type test_tensor_mask_highbits,@function
test_tensor_mask_highbits:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  s_or_b32 s4, s4, 0x7500000
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
.Ltest_tensor_mask_highbits_end:
.size test_tensor_mask_highbits, .Ltest_tensor_mask_highbits_end-test_tensor_mask_highbits

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_highbits
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_highbits
      .symbol: test_tensor_mask_highbits.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
