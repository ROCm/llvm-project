// COM: Two far sites each have the same sole 20-byte gateway. The earlier
// COM: site's greedy relay route would consume four bytes from that gateway
// COM: even though it has a separate relay route. Protect the later unfinished
// COM: owner, route the first site around the shared slot, then assign the
// COM: intact gateway to the second site.

// RUN: %clang -x assembler-with-cpp -DGATEWAY_BYTES=20 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: assigned 1 SCC-neutral forward gateway(s)
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// A 24-byte shared sled may donate one relay dword while preserving its one
// complete 20-byte gateway slot. This pins the exact capacity boundary and
// prevents reservation handling from excluding the whole sled.
// RUN: %clang -x assembler-with-cpp -DGATEWAY_BYTES=24 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.24.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.24.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.24.out.elf 2>&1 | %FileCheck --check-prefix=CAPACITY %s
// CAPACITY: hotswap: assigned 1 SCC-neutral forward gateway(s)
// CAPACITY: hotswap: assigned 1 forward s_branch island chain(s)
// CAPACITY-NOT: no safe short-branch gateway
// CAPACITY: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
#ifndef GATEWAY_BYTES
#define GATEWAY_BYTES 20
#endif
.text
.globl first_tied_site
.p2align 8
.type first_tied_site,@function
first_tied_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size first_tied_site, .-first_tied_site

.rept 19996
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

.type shared_sole_gateway,@function
shared_sole_gateway:
  s_endpgm
.size shared_sole_gateway, .-shared_sole_gateway
.fill GATEWAY_BYTES, 1, 0

.rept (50000 - GATEWAY_BYTES) / 4
  s_mov_b32 s64, s65
.endr

.globl second_tied_site
.type second_tied_site,@function
second_tied_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size second_tied_site, .-second_tied_site

.rept 7496
  s_mov_b32 s64, s65
.endr

.type final_relay,@function
final_relay:
  s_endpgm
.size final_relay, .-final_relay
.fill 4, 1, 0

.rept 30000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel first_tied_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel second_tied_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: first_tied_site
      .symbol: first_tied_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: second_tied_site
      .symbol: second_tied_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
