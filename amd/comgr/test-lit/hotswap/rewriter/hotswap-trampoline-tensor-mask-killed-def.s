// COM: A killed descriptor definition must not be treated as the reaching one.
// COM: The base is written by s_mov s4, 1 (an unrecognized value) after its
// COM: construction region, so the value reaching the tensor is not the
// COM: normalized descriptor. The definition clear is not applicable and the
// COM: pass falls back to the at-site rewrite.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: s4 dead, no save/restore needed
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_killed_def>:
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_killed_def
.p2align 8
.type test_tensor_killed_def,@function
test_tensor_killed_def:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  s_mov_b32 s4, 1
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
.Ltest_tensor_killed_def_end:
.size test_tensor_killed_def, .Ltest_tensor_killed_def_end-test_tensor_killed_def

.rodata
.p2align 8
.amdhsa_kernel test_tensor_killed_def
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_killed_def
      .symbol: test_tensor_killed_def.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
