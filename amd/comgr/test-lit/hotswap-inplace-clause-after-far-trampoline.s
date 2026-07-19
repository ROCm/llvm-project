// COM: A required far trampoline makes the following original text CFG-dead
// COM: after its source edge is installed. The blanket gfx1250 A0 clause
// COM: workaround still removes hard clauses in that original text.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: coalesced 2 adjacent ds_2addr rewrites at 0x0
// LOG: growWithTrampolines: appended 1 trampoline
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_clause_after_far_trampoline>:
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// COM: The coalesced source window moves the adjacent s_mov into the far
// COM: replacement body and pads its linked slot before resuming at global_wb.
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NOT: s_add_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_clause_after_far_trampoline
.p2align 8
.type test_clause_after_far_trampoline,@function
test_clause_after_far_trampoline:
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  s_mov_b32 s20, s21
  global_wb scope:SCOPE_CU
  v_nop
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
  .rept 40000
    s_mov_b32 s20, s21
  .endr
.Ltest_clause_after_far_trampoline_end:
.size test_clause_after_far_trampoline, .Ltest_clause_after_far_trampoline_end-test_clause_after_far_trampoline

.rodata
.p2align 8
.amdhsa_kernel test_clause_after_far_trampoline
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_clause_after_far_trampoline
      .symbol: test_clause_after_far_trampoline.kd
      .sgpr_count: 24
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
