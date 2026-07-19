// COM: An unresolved s_swap_pc_i64 that writes the ABI link pair is an opaque
// COM: call to a callable entry. It does not make arbitrary code padding a
// COM: possible target, so a separately proven far-branch gateway remains
// COM: usable. Nonstandard swaps, arbitrary set-PC transfers, and a statically
// COM: resolved standard-link call to a non-entry target still fail closed.

// RUN: %clang -x assembler-with-cpp -DCASE=0 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.standard.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.standard.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.standard.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=STANDARD %s
// STANDARD: hotswap: recognized ABI standard-link indirect call
// STANDARD: hotswap: assigned 1 SCC-neutral forward gateway(s)
// STANDARD: RESULT: SUCCESS

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.nonstandard.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nonstandard.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=POISON %s

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.setpc.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.setpc.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=POISON %s

// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.nonentry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nonentry.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NONENTRY %s

// POISON: hotswap: incomplete control-flow targets disable NOP padding donation
// POISON: hotswap: error: no safe short-branch gateway for far site
// POISON: RESULT: ERROR
// NONENTRY: materialized call to a non-entry target
// NONENTRY: hotswap: incomplete control-flow targets disable NOP padding donation
// NONENTRY: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl gateway_kernel
.type gateway_kernel,@function
gateway_kernel:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0
  s_endpgm
.size gateway_kernel, .-gateway_kernel

// This is the sole source-side set-PC gateway for the far patch above.
.rept 8
  s_nop 0
.endr

.local indirect_caller
.type indirect_caller,@function
indirect_caller:
#if CASE == 0
  s_swap_pc_i64 s[30:31], s[2:3]
#elif CASE == 1
  s_swap_pc_i64 s[28:29], s[2:3]
#elif CASE == 2
  s_set_pc_i64 s[2:3]
#elif CASE == 3
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], 12
  s_swap_pc_i64 s[30:31], s[2:3]
  s_endpgm
.Lnonentry:
#endif
  s_endpgm
.size indirect_caller, .-indirect_caller

.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel gateway_kernel
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: gateway_kernel
      .symbol: gateway_kernel.kd
      .sgpr_count: 66
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
