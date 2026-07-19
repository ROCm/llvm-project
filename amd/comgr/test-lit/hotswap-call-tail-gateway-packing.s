// COM: Three 8-byte far sites can reach only one 24-byte, one 16-byte, and one
// COM: 8-byte external gateway sled. The source-relative 12-byte call tails pack
// COM: twice into the 24-byte sled and once into the 16-byte sled. A full
// COM: get-PC/add/set-PC gateway could serve only two sites.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: assigned 3 SCC-neutral forward gateway(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d --mcpu=gfx1250 %t.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <call_tail_site0>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_nop 0
// DISASM-LABEL: <call_tail_site1>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_nop 0
// DISASM-LABEL: <call_tail_site2>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_nop 0
// DISASM-LABEL: <gateway_24>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-LABEL: <gateway_16>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl call_tail_site0
.p2align 8
.type call_tail_site0,@function
call_tail_site0:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size call_tail_site0, .-call_tail_site0

.rept 1021
  s_mov_b32 s64, s65
.endr

.globl call_tail_site1
.type call_tail_site1,@function
call_tail_site1:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size call_tail_site1, .-call_tail_site1

.rept 1021
  s_mov_b32 s64, s65
.endr

.globl call_tail_site2
.type call_tail_site2,@function
call_tail_site2:
  ds_load_2addr_stride64_b32 v[8:9], v10 offset0:1 offset1:3
  s_endpgm
.size call_tail_site2, .-call_tail_site2

.rept 4093
  s_mov_b32 s64, s65
.endr

.type gateway_24,@function
gateway_24:
  s_endpgm
.size gateway_24, .-gateway_24
.fill 24, 1, 0

.rept 1017
  s_mov_b32 s64, s65
.endr

.type gateway_16,@function
gateway_16:
  s_endpgm
.size gateway_16, .-gateway_16
.fill 16, 1, 0

.rept 1019
  s_mov_b32 s64, s65
.endr

.type gateway_8,@function
gateway_8:
  s_endpgm
.size gateway_8, .-gateway_8
.fill 8, 1, 0

.rept 70000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel call_tail_site0
  .amdhsa_next_free_vgpr 11
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel call_tail_site1
  .amdhsa_next_free_vgpr 11
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel call_tail_site2
  .amdhsa_next_free_vgpr 11
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: call_tail_site0
      .symbol: call_tail_site0.kd
      .sgpr_count: 66
      .vgpr_count: 11
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: call_tail_site1
      .symbol: call_tail_site1.kd
      .sgpr_count: 66
      .vgpr_count: 11
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: call_tail_site2
      .symbol: call_tail_site2.kd
      .sgpr_count: 66
      .vgpr_count: 11
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
