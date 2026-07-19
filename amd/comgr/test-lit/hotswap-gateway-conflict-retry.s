// COM: Both far sites share one 16-byte sled. Exact 12-byte call-tail accounting
// COM: leaves one dword for the earlier site's relay chain, avoiding the false
// COM: dependency cycle caused by the former 20-byte forward reservation.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: gateway reservation solver: {{[0-9]+}} protected pass(es),
// LOG-SAME: 0 conflict-directed retry/retries
// LOG-NOT: gateway allocation dependency cycle
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl first_conflicting_site
.p2align 8
.type first_conflicting_site,@function
first_conflicting_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size first_conflicting_site, .-first_conflicting_site

.rept 24996
  s_mov_b32 s64, s65
.endr

.type shared_gateway,@function
shared_gateway:
  s_endpgm
.size shared_gateway, .-shared_gateway
.fill 16, 1, 0

.rept 12496
  s_mov_b32 s64, s65
.endr

.globl second_conflicting_site
.type second_conflicting_site,@function
second_conflicting_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size second_conflicting_site, .-second_conflicting_site

// This position makes the first trampoline reachable from final_relay while
// the second, which follows it in the pool, is just out of branch range.
.rept 9157
  s_mov_b32 s64, s65
.endr

.type final_relay,@function
final_relay:
  s_endpgm
.size final_relay, .-final_relay
.fill 4, 1, 0

.rept 31162
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel first_conflicting_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel second_conflicting_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: first_conflicting_site
      .symbol: first_conflicting_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: second_conflicting_site
      .symbol: second_conflicting_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
