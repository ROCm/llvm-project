// COM: Reduced from corpus object
// COM: 8737ad6b480494d2a19aa82de8dfdd53e6c4b788975bd57e3c3437bc368ea780.
// COM: The original B0 ds_store_2addr_b64 uses element offsets (0, 1), which
// COM: become byte offsets (0, 8) in the canonical A0 split form.
// COM:
// COM: The unresolved register call deliberately keeps indirect entry points
// COM: unbounded, and the filler makes any appended pool unreachable by
// COM: s_branch. The old in-place shortcut bypassed those routing constraints.
// COM: With semantic splitting mandatory, fail closed on the unavailable
// COM: gateway instead of retaining an A0-unsafe DS2 instruction.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG-NOT: rewrote ds_store_2addr_b64
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR
// RUN: test ! -e %t.out.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_entry_local
.p2align 8
.type test_ds2_entry_local,@function
test_ds2_entry_local:
  s_swap_pc_i64 s[30:31], s[0:1]
  ds_store_2addr_b64 v19, v[14:15], v[20:21] offset1:1
  s_wait_loadcnt_dscnt 0x1
  s_endpgm
.size test_ds2_entry_local, .-test_ds2_entry_local

// Keep a hypothetical appended trampoline pool outside signed s_branch reach.
.rept 40000
  s_mov_b32 s2, s3
.endr

.rodata
.p2align 8
.amdhsa_kernel test_ds2_entry_local
  .amdhsa_next_free_vgpr 22
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2_entry_local
      .symbol: test_ds2_entry_local.kd
      .sgpr_count: 32
      .vgpr_count: 22
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
