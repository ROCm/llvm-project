// COM: A VMEM operation moved to an external HotSwap trampoline still counts
// COM: as prior VMEM after the trampoline returns. The blanket gfx1250 A0
// COM: clause workaround nevertheless removes the following hard clause.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_clause_after_vmem_trampoline>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_loadcnt 0x0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[6:7], off
// DISASM: s_mov_b32 [[SCRATCH:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b32 v4, v1, s[2:3]
// DISASM-NEXT: s_mov_b32 m0, [[SCRATCH]]
// DISASM-NEXT: s_branch
// DISASM-NOT: s_add_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_clause_after_vmem_trampoline
.p2align 8
.type test_clause_after_vmem_trampoline,@function
test_clause_after_vmem_trampoline:
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  s_clause 0x0
  global_load_b32 v0, v[6:7], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_clause_after_vmem_trampoline_end:
.size test_clause_after_vmem_trampoline, .Ltest_clause_after_vmem_trampoline_end-test_clause_after_vmem_trampoline

.rodata
.p2align 8
.amdhsa_kernel test_clause_after_vmem_trampoline
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_clause_after_vmem_trampoline
      .symbol: test_clause_after_vmem_trampoline.kd
      .sgpr_count: 16
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
