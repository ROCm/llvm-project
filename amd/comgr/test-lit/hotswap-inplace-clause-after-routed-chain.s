// COM: Reconstruct a HotSwap-style routed tail-cave chain whose body performs
// COM: no VMEM. The source branch reaches the clause through two external
// COM: direct-branch hops. Its straight-line body exceeds 64 KiB, which is a
// COM: legal size for a coalesced HotSwap body. The clause still contains the
// COM: kernel's first VMEM and must be removed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_clause_after_routed_chain>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: global_load_b32 v0, v[2:3], off
// DISASM: s_branch
// DISASM-NEXT: s_mov_b32 s4, s5
// DISASM: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_clause_after_routed_chain
.p2align 8
.type test_clause_after_routed_chain,@function
test_clause_after_routed_chain:
  s_branch .Lroute_forward
.Lroute_resume:
  s_clause 0x0
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_clause_after_routed_chain_end:
.size test_clause_after_routed_chain, .Ltest_clause_after_routed_chain_end-test_clause_after_routed_chain

.Lroute_forward:
  s_branch .Lroute_body
.Lroute_body:
  .rept 16385
    s_mov_b32 s4, s5
  .endr
  s_branch .Lroute_resume

.rodata
.p2align 8
.amdhsa_kernel test_clause_after_routed_chain
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_clause_after_routed_chain
      .symbol: test_clause_after_routed_chain.kd
      .sgpr_count: 8
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
