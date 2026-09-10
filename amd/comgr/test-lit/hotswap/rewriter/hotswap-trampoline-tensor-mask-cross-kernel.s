// COM: The construction region of one kernel must not be borrowed by a bare
// COM: tensor in another. Kernel A builds the descriptor; kernel B issues a
// COM: bare tensor on the same SGPR. The scan is confined to each function, so
// COM: B does not see A's mask-set and falls back to the at-site rewrite.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API-DAG: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API-DAG: hotswap: tensor_load_to_lds: s4 dead, no save/restore needed
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// COM: Kernel A: definition clear. Kernel B: at-site s_pack_hh, no borrowed def.
// DISASM-LABEL: <test_tensor_kernel_a>:
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM-LABEL: <test_tensor_kernel_b>:
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]

// COM: Idempotency: kernel A's definition clear and kernel B's at-site rewrite
// COM: both round-trip byte-equal on a second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_kernel_a
.p2align 8
.type test_tensor_kernel_a,@function
test_tensor_kernel_a:
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
.Ltest_tensor_kernel_a_end:
.size test_tensor_kernel_a, .Ltest_tensor_kernel_a_end-test_tensor_kernel_a

.globl test_tensor_kernel_b
.p2align 8
.type test_tensor_kernel_b,@function
test_tensor_kernel_b:
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
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_kernel_b_end:
.size test_tensor_kernel_b, .Ltest_tensor_kernel_b_end-test_tensor_kernel_b

.rodata
.p2align 8
.amdhsa_kernel test_tensor_kernel_a
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_kernel_b
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_kernel_a
      .symbol: test_tensor_kernel_a.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_tensor_kernel_b
      .symbol: test_tensor_kernel_b.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
