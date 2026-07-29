// COM: A protected constant-slot function table is loader-relocated before
// COM: execution and made immutable by GNU_RELRO. Prove the table target and
// COM: the canonical load/call sequence so the public B0-to-A0 rewrite does
// COM: not treat the register call as object-wide unresolved control flow.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: (%llvm-readelf -l %t.elf; %llvm-readelf -r %t.elf) \
// RUN:   | %FileCheck --check-prefix=INPUT %s
// INPUT: GNU_RELRO
// INPUT: R_AMDGPU_RELATIVE64

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: resolved relocation-table call at 0x{{[0-9A-F]+}} to 1 finite target(s)
// LOG-NOT: hotswap: unresolved call target
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <table_helper>:
// DISASM: s_set_pc_i64 s[30:31]
// DISASM-LABEL: <table_kernel>:
// DISASM: s_swap_pc_i64 s[30:31], s[0:1]
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: ds_load_b32 v0
// DISASM-NEXT: ds_load_b32 v1

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl table_helper
.protected table_helper
.type table_helper,@function
table_helper:
  s_nop 0
  s_set_pc_i64 s[30:31]
.Ltable_helper_end:
.size table_helper, .Ltable_helper_end-table_helper

.globl table_kernel
.p2align 8
.type table_kernel,@function
table_kernel:
  s_get_pc_i64 s[54:55]
.Ltable_kernel_after_getpc:
  s_add_nc_u64 s[54:55], s[54:55], helper_table-.Ltable_kernel_after_getpc
  s_load_b64 s[0:1], s[54:55], 0
  s_wait_kmcnt 0
  s_swap_pc_i64 s[30:31], s[0:1]
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0
  s_endpgm
.Ltable_kernel_end:
.size table_kernel, .Ltable_kernel_end-table_kernel

.section .data.rel.ro,"aw",@progbits
.p2align 3
.globl helper_table
.protected helper_table
.type helper_table,@object
helper_table:
  .quad table_helper
.size helper_table, 8

.rodata
.p2align 8
.amdhsa_kernel table_kernel
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 56
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: table_kernel
      .symbol: table_kernel.kd
      .sgpr_count: 56
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
