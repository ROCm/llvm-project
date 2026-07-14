// COM: Removing an unsafe hard clause must also release its members from the
// COM: clause's relocation protection. The SGPR-relative cluster load needs a
// COM: relocating A0 mask rewrite, so retaining stale clause protection would
// COM: make this otherwise valid object fail.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_clause_release>:
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCRATCH:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b32 v4, v1, s[2:3]
// DISASM-NEXT: s_mov_b32 m0, [[SCRATCH]]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_clause_release
.p2align 8
.type test_clause_release,@function
test_clause_release:
  s_clause 0x0
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  s_endpgm
  .rept 24
    s_nop 0
  .endr
.Ltest_clause_release_end:
.size test_clause_release, .Ltest_clause_release_end-test_clause_release

.rodata
.p2align 8
.amdhsa_kernel test_clause_release
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_clause_release
      .symbol: test_clause_release.kd
      .sgpr_count: 4
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
