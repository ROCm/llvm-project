// COM: Two descriptors may name different entries inside one sized STT_FUNC.
// COM: The interior descriptor entry starts a fresh initial-VMEM path even
// COM: though the outer entry executes VMEM first.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: %llvm-readelf -s %t.elf | %FileCheck --check-prefix=SYMBOLS %s
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

// SYMBOLS-DAG: FUNC{{.*}}descriptor_range_owner
// SYMBOLS-DAG: NOTYPE{{.*}}descriptor_range_interior

// DISASM-LABEL: <descriptor_range_interior>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl descriptor_range_owner
.p2align 8
.type descriptor_range_owner,@function
descriptor_range_owner:
  global_wb scope:SCOPE_CU
  v_nop
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0
  s_endpgm

.globl descriptor_range_interior
.p2align 8
.type descriptor_range_interior,@notype
descriptor_range_interior:
  s_clause 0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0
  s_endpgm
.Ldescriptor_range_end:
.size descriptor_range_owner, .Ldescriptor_range_end-descriptor_range_owner

.rodata
.p2align 8
.amdhsa_kernel descriptor_range_owner
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel descriptor_range_interior
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: descriptor_range_owner
      .symbol: descriptor_range_owner.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: descriptor_range_interior
      .symbol: descriptor_range_interior.kd
      .sgpr_count: 0
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
