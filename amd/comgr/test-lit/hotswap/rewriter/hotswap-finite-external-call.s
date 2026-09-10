// COM: A canonical PC-materialized call with one target outside .text has no
// COM: local destination, but its continuation is a local entry. Proving both
// COM: facts permits a far required rewrite to use the code-end gateway while
// COM: preserving the external call's return point.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: resolved PC-materialized call
// LOG-SAME: to finite external target
// LOG-NOT: hotswap: unresolved call target
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf \
// RUN:   | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <finite_external_call>:
// DISASM: s_swap_pc_i64
// DISASM-NEXT: s_branch
// DISASM: ds_load_b32 v0, v2 offset:256
// DISASM-NEXT: ds_load_b32 v1, v2 offset:768

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl finite_external_call
.p2align 8
.type finite_external_call,@function
finite_external_call:
.Lgetpc:
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], 0x100000000
  s_swap_pc_i64 s[6:7], s[2:3]
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.size finite_external_call, .-finite_external_call

.fill 20, 1, 0
.rept 40000
  s_mov_b32 s10, s11
.endr

.rodata
.p2align 8
.amdhsa_kernel finite_external_call
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: finite_external_call
      .symbol: finite_external_call.kd
      .sgpr_count: 12
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
