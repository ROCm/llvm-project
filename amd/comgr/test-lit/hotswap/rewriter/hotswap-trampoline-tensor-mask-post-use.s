// COM: A non-tensor consumer after the tensor still observes a definition-time
// COM: clear. Reject that path and use the at-site save/restore fallback so the
// COM: later consumer sees the original descriptor value.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: s4 live, save/restore via s{{[0-9]+}}
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_post_use>:
// DISASM: s_and_b32 s4, s4, 0xfff7ffff
// DISASM-NOT: s_and_b32 s4, s4, 0xfff70000
// DISASM: s_mov_b32 s0, s4
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_mov_b32 s4, s{{[0-9]+}}

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_post_use
.p2align 8
.type test_tensor_post_use,@function
test_tensor_post_use:
  s_mov_b32 s4, 0
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
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
.Ltest_tensor_post_use_end:
.size test_tensor_post_use, .Ltest_tensor_post_use_end-test_tensor_post_use

.rodata
.p2align 8
.amdhsa_kernel test_tensor_post_use
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_post_use
      .symbol: test_tensor_post_use.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
