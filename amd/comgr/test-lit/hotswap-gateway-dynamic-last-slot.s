// COM: The later site starts with two complete gateways. The earlier relay may
// COM: consume four bytes from the first (2 -> 1), but when its next greedy hop
// COM: would consume the second (1 -> 0), it must take an alternate relay.
// COM: The later site can then use the one intact 20-byte gateway.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: gateway reservation solver:
// LOG: hotswap: assigned 1 SCC-neutral forward gateway(s)
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl first_dynamic_site
.p2align 8
.type first_dynamic_site,@function
first_dynamic_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size first_dynamic_site, .-first_dynamic_site

.rept 24996
  s_mov_b32 s64, s65
.endr

.type first_full_gateway,@function
first_full_gateway:
  s_endpgm
.size first_full_gateway, .-first_full_gateway
.fill 20, 1, 0

.rept 12495
  s_mov_b32 s64, s65
.endr

.globl later_dynamic_site
.type later_dynamic_site,@function
later_dynamic_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size later_dynamic_site, .-later_dynamic_site

.rept 7496
  s_mov_b32 s64, s65
.endr

.type alternate_relay,@function
alternate_relay:
  s_endpgm
.size alternate_relay, .-alternate_relay
.fill 4, 1, 0

.rept 4998
  s_mov_b32 s64, s65
.endr

.type second_full_gateway,@function
second_full_gateway:
  s_endpgm
.size second_full_gateway, .-second_full_gateway
.fill 20, 1, 0

.rept 24994
  s_mov_b32 s64, s65
.endr

.type final_relay,@function
final_relay:
  s_endpgm
.size final_relay, .-final_relay
.fill 4, 1, 0

.rept 25000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel first_dynamic_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel later_dynamic_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: first_dynamic_site
      .symbol: first_dynamic_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: later_dynamic_site
      .symbol: later_dynamic_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
