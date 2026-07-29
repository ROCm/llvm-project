// COM: Verify that the LLVM MC legacy gfx1250 tensor records are consumed by
// COM: the existing HotSwap tensor path. Loads use the established descriptor
// COM: mask wrapper. Stores retain one 12-byte boundary and their canonical
// COM: scalar operands, but do not receive the load-only wrapper.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: %llvm-objdump -d %t.elf | %FileCheck --check-prefix=INPUT %s
// INPUT-LABEL: <legacy_load_d2>:
// INPUT: tensor_load_to_lds s[0:3], s[4:11]
// INPUT-NEXT: s_mov_b32 s0, s4
// INPUT-LABEL: <legacy_load_d4>:
// INPUT: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19]
// INPUT-NEXT: s_endpgm
// INPUT-LABEL: <legacy_load_saturated>:
// INPUT: tensor_load_to_lds s[0:3], s[4:11]
// INPUT-NEXT: s_mov_b32 s0, s4
// INPUT-LABEL: <legacy_store_d2>:
// INPUT: tensor_store_from_lds s[0:3], s[4:11]
// INPUT-NEXT: s_mov_b32 s0, s4
// INPUT-NEXT: s_endpgm
// INPUT-LABEL: <legacy_store_d4>:
// INPUT: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19]
// INPUT-NEXT: s_mov_b32 s0, s4
// INPUT-NEXT: s_endpgm

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: s4 live, save/restore via s{{[0-9]+}}
// API: hotswap: tensor_load_to_lds: s4 dead, no save/restore needed
// API: hotswap: tensor_load_to_lds: reusing locally dead s103
// API: hotswap: tensor_load_to_lds: s4 live, save/restore via s103
// API-NOT: tensor_store_from_lds
// API: RESULT: SUCCESS

// RUN: %llvm-readelf -h %t.out.elf | %FileCheck --check-prefix=ELF %s
// ELF: Class:                             ELF64
// ELF: Machine:                           EM_AMDGPU

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=OUTPUT %s
// OUTPUT-LABEL: <legacy_load_d2>:
// OUTPUT: s_branch
// OUTPUT: s_mov_b32 s0, s4
// OUTPUT: s_endpgm
// OUTPUT: s_mov_b32 [[SCRATCH:s[0-9]+]], s4
// OUTPUT-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// OUTPUT-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// OUTPUT-NEXT: s_mov_b32 s4, [[SCRATCH]]
// OUTPUT-NEXT: s_branch

// OUTPUT-LABEL: <legacy_load_d4>:
// OUTPUT: s_branch
// OUTPUT: s_endpgm
// OUTPUT: s_pack_hh_b32_b16 s4, 0, s4
// OUTPUT-NEXT: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19]
// OUTPUT-NEXT: s_branch

// OUTPUT-LABEL: <legacy_load_saturated>:
// OUTPUT: s_branch
// OUTPUT: s_mov_b32 s0, s4
// OUTPUT: s_endpgm
// OUTPUT: s_mov_b32 s103, s4
// OUTPUT-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// OUTPUT-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// OUTPUT-NEXT: s_mov_b32 s4, s103
// OUTPUT-NEXT: s_branch

// OUTPUT-LABEL: <legacy_store_d2>:
// OUTPUT-NOT: s_pack_hh_b32_b16
// OUTPUT-NOT: s_branch
// OUTPUT: tensor_store_from_lds s[0:3], s[4:11]
// OUTPUT-NEXT: s_mov_b32 s0, s4
// OUTPUT-NEXT: s_endpgm

// OUTPUT-LABEL: <legacy_store_d4>:
// OUTPUT-NOT: s_pack_hh_b32_b16
// OUTPUT-NOT: s_branch
// OUTPUT: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19]
// OUTPUT-NEXT: s_mov_b32 s0, s4
// OUTPUT-NEXT: s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl legacy_load_d2
.p2align 8
.type legacy_load_d2,@function
legacy_load_d2:
  tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11]
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
.Llegacy_load_d2_end:
.size legacy_load_d2, .Llegacy_load_d2_end-legacy_load_d2

.globl legacy_load_d4
.p2align 8
.type legacy_load_d4,@function
legacy_load_d4:
  tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19]
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
.Llegacy_load_d4_end:
.size legacy_load_d4, .Llegacy_load_d4_end-legacy_load_d4

.globl legacy_load_saturated
.p2align 8
.type legacy_load_saturated,@function
legacy_load_saturated:
  tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11]
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
.Llegacy_load_saturated_end:
.size legacy_load_saturated, .Llegacy_load_saturated_end-legacy_load_saturated

.globl legacy_store_d2
.p2align 8
.type legacy_store_d2,@function
legacy_store_d2:
  tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
.Llegacy_store_d2_end:
.size legacy_store_d2, .Llegacy_store_d2_end-legacy_store_d2

.globl legacy_store_d4
.p2align 8
.type legacy_store_d4,@function
legacy_store_d4:
  tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19]
  s_mov_b32 s0, s4
  s_endpgm
.Llegacy_store_d4_end:
.size legacy_store_d4, .Llegacy_store_d4_end-legacy_store_d4

.rodata
.p2align 8
.amdhsa_kernel legacy_load_d2
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel legacy_load_d4
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 20
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel legacy_load_saturated
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel legacy_store_d2
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel legacy_store_d4
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 20
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: legacy_load_d2
      .symbol: legacy_load_d2.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: legacy_load_d4
      .symbol: legacy_load_d4.kd
      .sgpr_count: 20
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: legacy_load_saturated
      .symbol: legacy_load_saturated.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: legacy_store_d2
      .symbol: legacy_store_d2.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: legacy_store_d4
      .symbol: legacy_store_d4.kd
      .sgpr_count: 20
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
