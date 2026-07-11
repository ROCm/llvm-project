// COM: gfx1250 A0 supports hard clauses when every member has the same cache
// COM: scope. This matches the AITER act_and_mul pattern observed in DSv4 Pro:
// COM: two TH_LOAD_LU buffer loads covered by one clause. HotSwap must retain
// COM: the marker instead of applying the mixed-scope workaround globally.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_uniform_clause>:
// DISASM-NEXT: s_clause 0x1
// DISASM-NEXT: buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
// DISASM-NEXT: buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU
// DISASM-NEXT: s_wait_loadcnt 0x0

// COM: A second rewrite must retain the same clause and be byte-identical.
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
  s_clause 0x1
  buffer_load_b128 v[0:3], v18, s[4:7], null offen th:TH_LOAD_LU
  buffer_load_b128 v[4:7], v18, s[8:11], null offen th:TH_LOAD_LU
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_uniform_clause_end:
.size test_uniform_clause, .Ltest_uniform_clause_end-test_uniform_clause

.rodata
.p2align 8
.amdhsa_kernel test_uniform_clause
  .amdhsa_next_free_vgpr 19
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
