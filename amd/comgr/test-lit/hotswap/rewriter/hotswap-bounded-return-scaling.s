// COM: Exercise bounded-return analysis with enough local functions and
// COM: decoded instructions to expose repeated whole-object scans.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf --entry-trampolines --strict-mode \
// RUN:   | %FileCheck %s
// CHECK: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl bounded_return_scaling
.p2align 8
.type bounded_return_scaling,@function
bounded_return_scaling:
  global_wb
  v_nop
  s_endpgm
.size bounded_return_scaling, .-bounded_return_scaling

// s_endpgm proves that the next helper cannot fall through from its
// predecessor. Each helper then reaches the function-local link-register
// proof that previously scanned the complete decoded stream.
.macro EMIT_LOCAL_HELPER
  .type bounded_return_helper_\@,@function
bounded_return_helper_\@:
  .rept 200
    s_nop 0
  .endr
  s_set_pc_i64 s[30:31]
  .size bounded_return_helper_\@, .-bounded_return_helper_\@
  s_endpgm
.endm

.rept 1000
  EMIT_LOCAL_HELPER
.endr

.rodata
.p2align 8
.amdhsa_kernel bounded_return_scaling
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: bounded_return_scaling
      .symbol: bounded_return_scaling.kd
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
