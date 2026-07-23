// COM: A0 tensor mask clearing fails closed when the descriptor base is written
// COM: by an instruction outside the recognized construction idiom (here an
// COM: s_add after the mask-set). The value reaching the tensor is then not the
// COM: normalized descriptor whose low16 the pass clears, so it rejects.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds at 0x{{[0-9A-F]+}}: descriptor base s4 is written by an unrecognized instruction at 0x{{[0-9A-F]+}} (s_add_co_u32)
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_mutated
.p2align 8
.type test_tensor_mask_mutated,@function
test_tensor_mask_mutated:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfff7ffff
  s_add_u32 s4, s4, 1
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_mask_mutated_end:
.size test_tensor_mask_mutated, .Ltest_tensor_mask_mutated_end-test_tensor_mask_mutated

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_mutated
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_mutated
      .symbol: test_tensor_mask_mutated.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
