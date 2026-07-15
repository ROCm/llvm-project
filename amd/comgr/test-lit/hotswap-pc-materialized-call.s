// COM: A canonical compiler-emitted register call materializes its target as
// COM: s_get_pc_i64 plus s_add_nc_u64. Prove that target from the decoded MC
// COM: operands instead of treating the call as globally unresolved. The
// COM: target is the second adjacent far patch site, which must remain an
// COM: independently callable entry while safe external gateways remain
// COM: available for both patches.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: resolved PC-materialized call at 0x8 to .text+0x14
// LOG-NOT: hotswap: unresolved call target
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_pc_materialized_call>:
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3], s[2:3], 16
// DISASM-NEXT: s_swap_pc_i64 s[0:1], s[2:3]
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch

// COM: A second rewrite must preserve the resolved call and patched object.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_pc_materialized_call
.p2align 8
.type test_pc_materialized_call,@function
test_pc_materialized_call:
  // The captured PC is .text+4; adding 16 selects .text+20, the second DS
  // instruction below.
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], 16
  s_swap_pc_i64 s[0:1], s[2:3]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lpc_materialized_target:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_pc_materialized_call_end:
.size test_pc_materialized_call, .Ltest_pc_materialized_call_end-test_pc_materialized_call

// Each far site needs its own 28-byte SCC-preserving gateway. This padding
// follows s_endpgm and lies outside the function, so it is safe.
.fill 64, 1, 0

// Push the appended trampoline pool beyond s_branch's signed 16-bit dword
// range and force direct-target-aware far-site handling.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_pc_materialized_call
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_pc_materialized_call
      .symbol: test_pc_materialized_call.kd
      .sgpr_count: 4
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
