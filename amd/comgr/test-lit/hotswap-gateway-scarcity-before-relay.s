// COM: Two far sites share one gateway sled. The earlier site can also use a
// COM: separate 12-byte call-tail gateway, but its greedy relay route crosses
// COM: the shared sled first. Process the later, one-slot site first; it takes
// COM: the other relay route and leaves the earlier site its alternate gateway.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: assigned 1 SCC-neutral forward gateway(s)
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG-NOT: no safe short-branch gateway
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <abundant_site>:
// DISASM-NEXT: s_call_i64
// DISASM-LABEL: <scarce_site>:
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl abundant_site
.p2align 8
.type abundant_site,@function
abundant_site:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size abundant_site, .-abundant_site

// This call-tail gateway is reachable only from the earlier site.
.type abundant_gateway,@function
abundant_gateway:
  s_endpgm
.size abundant_gateway, .-abundant_gateway
.fill 20, 1, 0

// Put the shared gateway about 100 KB from the first source.
.rept 24990
  s_mov_b32 s64, s65
.endr

.type shared_gateway,@function
shared_gateway:
  s_endpgm
.size shared_gateway, .-shared_gateway
.fill 20, 1, 0

// The second source is about 150 KB from the start. Its only gateway sled is
// shared_gateway: abundant_gateway is out of short-branch range and the later
// relay has only one dword.
.rept 12495
  s_mov_b32 s64, s65
.endr

.globl scarce_site
.type scarce_site,@function
scarce_site:
  ds_load_2addr_stride64_b32 v[4:5], v6 offset0:1 offset1:3
  s_endpgm
.size scarce_site, .-scarce_site

.rept 12497
  s_mov_b32 s64, s65
.endr

.type relay_gateway,@function
relay_gateway:
  s_endpgm
.size relay_gateway, .-relay_gateway
.fill 4, 1, 0

// Keep both appended trampolines far from their source sites. Before gateway
// reservation, abundant_site relayed through shared_gateway and relay_gateway,
// consuming the only full slot available to scarce_site.
.rept 25000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel abundant_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel scarce_site
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: abundant_site
      .symbol: abundant_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: scarce_site
      .symbol: scarce_site.kd
      .sgpr_count: 66
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
