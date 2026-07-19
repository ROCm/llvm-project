// COM: The first site's farthest relay consumes one of the second site's two
// COM: full gateway slots, then its only next hop would consume the last one.
// COM: Backtrack to the nearer relay first; the later hop then leaves the
// COM: second site's other full gateway intact.

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
.globl relay_search_site
.p2align 8
.type relay_search_site,@function
relay_search_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size relay_search_site, .-relay_search_site

.rept 19996
  s_mov_b32 s64, s65
.endr

.type nearer_relay,@function
nearer_relay:
  s_endpgm
.size nearer_relay, .-nearer_relay
.fill 4, 1, 0

.rept 4998
  s_mov_b32 s64, s65
.endr

.type farthest_gateway,@function
farthest_gateway:
  s_endpgm
.size farthest_gateway, .-farthest_gateway
.fill 20, 1, 0

.rept 12495
  s_mov_b32 s64, s65
.endr

.globl reserved_site
.type reserved_site,@function
reserved_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size reserved_site, .-reserved_site

.rept 6085
  s_mov_b32 s64, s65
.endr

.type later_gateway,@function
later_gateway:
  s_endpgm
.size later_gateway, .-later_gateway
.fill 20, 1, 0

.rept 31411
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel relay_search_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel reserved_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: relay_search_site
      .symbol: relay_search_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: reserved_site
      .symbol: reserved_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
