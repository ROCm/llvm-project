// COM: Exercise the public rewrite path with thousands of unrelated local
// COM: set-PC functions. Control-flow proof must remain fail-closed without
// COM: forming the Cartesian product of every set-PC and function range.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-readelf --file-header --section-headers --program-headers \
// RUN:   %t.out.elf > /dev/null

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.macro local_set_pc_function
.type local_set_pc_\@,@function
local_set_pc_\@:
  s_set_pc_i64 s[0:1]
.Llocal_set_pc_end_\@:
.size local_set_pc_\@, .Llocal_set_pc_end_\@-local_set_pc_\@
.endm

.rept 4096
  local_set_pc_function
.endr

.globl control_flow_index_kernel
.protected control_flow_index_kernel
.type control_flow_index_kernel,@function
control_flow_index_kernel:
  s_endpgm
.Lcontrol_flow_index_kernel_end:
.size control_flow_index_kernel, .Lcontrol_flow_index_kernel_end-control_flow_index_kernel

.rodata
.p2align 8
.amdhsa_kernel control_flow_index_kernel
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: control_flow_index_kernel
      .symbol: control_flow_index_kernel.kd
      .sgpr_count: 0
      .vgpr_count: 0
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
