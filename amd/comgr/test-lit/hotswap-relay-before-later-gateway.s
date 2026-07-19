// COM: A relay-only site needs one dword from a 20-byte sled that is also the
// COM: other site's sole full gateway. Do not globally allocate every full
// COM: gateway: the one-slot site takes its independent relay route first,
// COM: leaving the shared sled available to the relay-only chain.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG-NOT: SCC-neutral forward gateway
// LOG: hotswap: assigned 2 forward s_branch island chain(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl relay_only_site
.p2align 8
.type relay_only_site,@function
relay_only_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size relay_only_site, .-relay_only_site

.rept 24996
  s_mov_b32 s64, s65
.endr

.type first_relay,@function
first_relay:
  s_endpgm
.size first_relay, .-first_relay
.fill 4, 1, 0

.rept 12499
  s_mov_b32 s64, s65
.endr

.globl reservable_site
.type reservable_site,@function
reservable_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size reservable_site, .-reservable_site

.rept 12496
  s_mov_b32 s64, s65
.endr

// relay_only_site cannot reach this sled directly, so it initially has zero
// full gateway candidates. It reaches the sled through first_relay instead.
.type shared_full_gateway,@function
shared_full_gateway:
  s_endpgm
.size shared_full_gateway, .-shared_full_gateway
.fill 20, 1, 0

.rept 19994
  s_mov_b32 s64, s65
.endr

// reservable_site is processed first and uses this independent route instead
// of allocating shared_full_gateway.
.type reservable_relay,@function
reservable_relay:
  s_endpgm
.size reservable_relay, .-reservable_relay
.fill 4, 1, 0

.rept 7498
  s_mov_b32 s64, s65
.endr

.type relay_only_final,@function
relay_only_final:
  s_endpgm
.size relay_only_final, .-relay_only_final
.fill 4, 1, 0

.rept 22500
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel relay_only_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel reservable_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: relay_only_site
      .symbol: relay_only_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: reservable_site
      .symbol: reservable_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
