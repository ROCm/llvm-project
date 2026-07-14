// COM: Removing an unsafe clause must not release a member that is also in an
// COM: s_delay_alu dependency span. The cluster-load rewrite cannot relocate
// COM: this protected member, so HotSwap must continue to fail closed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: inplace: initial-VMEM s_clause -> s_nop
// LOG: hotswap: error: replacement source at 0x8 is relocation-protected
// LOG: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_clause_delay_overlap
.p2align 8
.type test_clause_delay_overlap,@function
test_clause_delay_overlap:
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
  s_clause 0x0
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  s_endpgm
  .rept 24
    s_nop 0
  .endr
.Ltest_clause_delay_overlap_end:
.size test_clause_delay_overlap, .Ltest_clause_delay_overlap_end-test_clause_delay_overlap

.rodata
.p2align 8
.amdhsa_kernel test_clause_delay_overlap
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_clause_delay_overlap
      .symbol: test_clause_delay_overlap.kd
      .sgpr_count: 4
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
