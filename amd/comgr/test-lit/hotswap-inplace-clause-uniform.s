// COM: The gfx1250 A0 workaround removes every hard clause, including clauses
// COM: whose members have a uniform cache scope and clauses reached only after
// COM: prior VMEM. Verify that each clause becomes an equal-sized scalar NOP.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_uniform_clause>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
// DISASM-NEXT: buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU

// DISASM-NEXT: s_wait_loadcnt 0x0

// COM: A second rewrite must retain the replacement NOP and be byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_uniform_clause
.p2align 8
.type test_uniform_clause,@function
test_uniform_clause:
  global_wb scope:SCOPE_CU
  v_nop
  s_clause 0x1
  buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
  buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_uniform_clause_end:
.size test_uniform_clause, .Ltest_uniform_clause_end-test_uniform_clause

// DISASM-LABEL: <test_initial_vmem_clause>:
// DISASM-NEXT: s_load_b64 s[0:1], s[0:1], 0x0 nv
// DISASM-NEXT: s_wait_kmcnt 0x0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b128 v[0:3], v[8:9], off
// DISASM-NEXT: global_load_b128 v[4:7], v[8:9], off offset:16
// DISASM-NEXT: global_load_b128 v[10:13], v[8:9], off offset:32
// DISASM-NEXT: global_load_b128 v[14:17], v[8:9], off offset:48

.globl test_initial_vmem_clause
.p2align 8
.type test_initial_vmem_clause,@function
test_initial_vmem_clause:
  s_load_b64 s[0:1], s[0:1], 0x0 nv
  s_wait_kmcnt 0x0
  s_clause 0x3
  global_load_b128 v[0:3], v[8:9], off
  global_load_b128 v[4:7], v[8:9], off offset:16
  global_load_b128 v[10:13], v[8:9], off offset:32
  global_load_b128 v[14:17], v[8:9], off offset:48
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_initial_vmem_clause_end:
.size test_initial_vmem_clause, .Ltest_initial_vmem_clause_end-test_initial_vmem_clause

// COM: Match the exact two-load AITER activation pattern as well as the
// COM: four-load quantization pattern above.
// DISASM-LABEL: <test_initial_buffer_clause>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
// DISASM-NEXT: buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU

.globl test_initial_buffer_clause
.p2align 8
.type test_initial_buffer_clause,@function
test_initial_buffer_clause:
  s_clause 0x1
  buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
  buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_initial_buffer_clause_end:
.size test_initial_buffer_clause, .Ltest_initial_buffer_clause_end-test_initial_buffer_clause

// A textually earlier VMEM does not satisfy the entry requirement when a
// branch can bypass it.
// DISASM-LABEL: <test_bypassed_prior_vmem>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1, v[2:3], off

.globl test_bypassed_prior_vmem
.p2align 8
.type test_bypassed_prior_vmem,@function
test_bypassed_prior_vmem:
  s_branch .Ltest_bypassed_prior_vmem_clause
  global_load_b32 v0, v[2:3], off
.Ltest_bypassed_prior_vmem_clause:
  s_clause 0x0
  global_load_b32 v1, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_bypassed_prior_vmem_end:
.size test_bypassed_prior_vmem, .Ltest_bypassed_prior_vmem_end-test_bypassed_prior_vmem

// Even when an exiting branch cannot bypass the prior VMEM, blanket clause
// removal replaces the clause.
// DISASM-LABEL: <test_exit_branch_prior_vmem>:
// DISASM-NEXT: s_cmp_eq_u32 s0, 0
// DISASM-NEXT: s_cbranch_scc1
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1, v[4:5], off

.globl test_exit_branch_prior_vmem
.p2align 8
.type test_exit_branch_prior_vmem,@function
test_exit_branch_prior_vmem:
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Ltest_exit_branch_prior_vmem_exit
  global_load_b32 v0, v[2:3], off
  s_clause 0x0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0x0
.Ltest_exit_branch_prior_vmem_exit:
  s_endpgm
.Ltest_exit_branch_prior_vmem_end:
.size test_exit_branch_prior_vmem, .Ltest_exit_branch_prior_vmem_end-test_exit_branch_prior_vmem

// Both arms of this diamond execute a VMEM before reaching the join. Blanket
// clause removal still replaces the clause.
// DISASM-LABEL: <test_diamond_all_prior_vmem>:
// DISASM-NEXT: s_cmp_eq_u32 s0, 0
// DISASM-NEXT: s_cbranch_scc1
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NEXT: s_branch
// DISASM-NEXT: global_load_b32 v2, v[2:3], off
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1, v[4:5], off

.globl test_diamond_all_prior_vmem
.p2align 8
.type test_diamond_all_prior_vmem,@function
test_diamond_all_prior_vmem:
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Ltest_diamond_all_prior_vmem_right
  global_load_b32 v0, v[2:3], off
  s_branch .Ltest_diamond_all_prior_vmem_join
.Ltest_diamond_all_prior_vmem_right:
  global_load_b32 v2, v[2:3], off
.Ltest_diamond_all_prior_vmem_join:
  s_clause 0x0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_diamond_all_prior_vmem_end:
.size test_diamond_all_prior_vmem, .Ltest_diamond_all_prior_vmem_end-test_diamond_all_prior_vmem

// One arm of this diamond reaches the join without a VMEM. The clause can
// therefore contain the first dynamically reached VMEM and must be removed.
// DISASM-LABEL: <test_diamond_missing_prior_vmem>:
// DISASM-NEXT: s_cmp_eq_u32 s0, 0
// DISASM-NEXT: s_cbranch_scc1
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_mov_b32 s1, 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v1, v[4:5], off

.globl test_diamond_missing_prior_vmem
.p2align 8
.type test_diamond_missing_prior_vmem,@function
test_diamond_missing_prior_vmem:
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Ltest_diamond_missing_prior_vmem_right
  global_load_b32 v0, v[2:3], off
  s_branch .Ltest_diamond_missing_prior_vmem_join
.Ltest_diamond_missing_prior_vmem_right:
  s_mov_b32 s1, 0
.Ltest_diamond_missing_prior_vmem_join:
  s_clause 0x0
  global_load_b32 v1, v[4:5], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_diamond_missing_prior_vmem_end:
.size test_diamond_missing_prior_vmem, .Ltest_diamond_missing_prior_vmem_end-test_diamond_missing_prior_vmem

// Kernel entry symbols may have st_size=0; the descriptor is authoritative.
// DISASM-LABEL: <test_zero_size_initial_clause>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
.globl test_zero_size_initial_clause
.p2align 8
.type test_zero_size_initial_clause,@function
test_zero_size_initial_clause:
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm

// A call target may contain the first VMEM even though the ABI continuation
// immediately exits. Model both the unknown call target and the fallthrough.
// DISASM-LABEL: <test_call_first_vmem>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off

.globl test_call_first_vmem
.p2align 8
.type test_call_first_vmem,@function
test_call_first_vmem:
  s_call_i64 s[0:1], .Ltest_call_first_vmem_callee
  s_endpgm
.Ltest_call_first_vmem_callee:
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_set_pc_i64 s[0:1]
.Ltest_call_first_vmem_end:
.size test_call_first_vmem, .Ltest_call_first_vmem_end-test_call_first_vmem

// A resumable debug trap does not prove that subsequent instructions are
// unreachable. The clause after it can still contain the first VMEM.
// DISASM-LABEL: <test_resumable_trap_first_vmem>:
// DISASM-NEXT: s_trap 3
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off

.globl test_resumable_trap_first_vmem
.p2align 8
.type test_resumable_trap_first_vmem,@function
test_resumable_trap_first_vmem:
  s_trap 3
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_resumable_trap_first_vmem_end:
.size test_resumable_trap_first_vmem, .Ltest_resumable_trap_first_vmem_end-test_resumable_trap_first_vmem

// Functions without a kernel descriptor have no known entry fact. A helper
// reached before its caller executes a VMEM must therefore be conservative.
// DISASM-LABEL: <test_uncovered_helper_caller>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <test_uncovered_helper>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off

.globl test_uncovered_helper_caller
.p2align 8
.type test_uncovered_helper_caller,@function
test_uncovered_helper_caller:
  s_call_i64 s[0:1], test_uncovered_helper
  s_endpgm
.Ltest_uncovered_helper_caller_end:
.size test_uncovered_helper_caller, .Ltest_uncovered_helper_caller_end-test_uncovered_helper_caller

.local test_uncovered_helper
.p2align 8
.type test_uncovered_helper,@function
test_uncovered_helper:
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_set_pc_i64 s[0:1]
.Ltest_uncovered_helper_end:
.size test_uncovered_helper, .Ltest_uncovered_helper_end-test_uncovered_helper

.rodata
.p2align 8
.amdhsa_kernel test_uniform_clause
  .amdhsa_next_free_vgpr 19
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_initial_vmem_clause
  .amdhsa_next_free_vgpr 18
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_initial_buffer_clause
  .amdhsa_next_free_vgpr 19
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_bypassed_prior_vmem
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_exit_branch_prior_vmem
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_diamond_all_prior_vmem
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_diamond_missing_prior_vmem
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_zero_size_initial_clause
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_call_first_vmem
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_resumable_trap_first_vmem
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_uncovered_helper_caller
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
