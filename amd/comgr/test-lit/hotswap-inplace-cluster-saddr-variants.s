// COM: Test HotSwap in-place B0->A0 conversion of SGPR-relative (_SADDR)
// COM: cluster_load instructions across every variant and operand modifier.
// COM: Each cluster_load_*_SADDR is rewritten in place to the matching
// COM: global_load_*_saddr (same size, identical operand layout), which
// COM: neutralizes the A0 multicast because global_load never reads
// COM: M0.wg_mask. The rewrite reserves no scratch SGPR, emits no trampoline
// COM: branch, and preserves each load's offset / scale_offset / th / scope
// COM: modifiers.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// COM: Every SADDR cluster load is converted in place to global_load_*_saddr
// COM: with its modifiers intact, in source order. No cluster_load survives,
// COM: and no M0 mask save/clear/restore or trampoline branch is emitted.
// DISASM-LABEL: <test_cluster_saddr_inplace>:
// DISASM-NOT: cluster_load
// DISASM: global_load_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NOT: cluster_load
// DISASM: global_load_async_to_lds_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NOT: cluster_load
// DISASM: global_load_b128 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}] offset:64 scale_offset th:TH_LOAD_NT_HT scope:SCOPE_DEV
// DISASM-NOT: cluster_load
// DISASM: global_load_async_to_lds_b8 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] offset:-64 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// DISASM-NOT: cluster_load
// DISASM: global_load_async_to_lds_b64 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] scale_offset th:TH_LOAD_BYPASS scope:SCOPE_SYS
// DISASM-NOT: cluster_load
// DISASM: global_load_async_to_lds_b128 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] offset:64
// DISASM-NOT: cluster_load
// DISASM: s_endpgm
// DISASM-NOT: cluster_load
// DISASM-NOT: s_pack_hh_b32_b16
// DISASM-NOT: s_branch

// COM: No scratch SGPR is reserved for the conversion, so sgpr_count is
// COM: unchanged from the kernel's declared value.
// METADATA: .name:           test_cluster_saddr_inplace
// METADATA: .sgpr_count:     16

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: The in-place conversion needs no scratch SGPR, so a kernel near the
// COM: SGPR maximum still rewrites successfully -- the "no free SGPR" hazard
// COM: of the former M0 mask workaround is gone.
// RUN: sed -e 's/.amdhsa_next_free_sgpr 16/.amdhsa_next_free_sgpr 106/' \
// RUN:     -e 's/.sgpr_count: 16/.sgpr_count: 106/' %s > %t.highsgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.highsgpr.s -o %t.highsgpr.elf
// RUN: hotswap-rewrite %t.highsgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.highsgpr.out.elf \
// RUN:   | %FileCheck --check-prefix=HIGH-SGPR %s
// HIGH-SGPR: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cluster_saddr_inplace
.p2align 8
.type test_cluster_saddr_inplace,@function
test_cluster_saddr_inplace:
  cluster_load_b64 v[0:1], v2, s[4:5]
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b32 v3, v4, s[6:7]
  s_wait_loadcnt 0x0
  cluster_load_b128 v[8:11], v12, s[8:9] offset:64 scale_offset th:TH_LOAD_NT_HT scope:SCOPE_DEV
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b8 v13, v14, s[10:11] offset:-64 th:TH_LOAD_NT_HT scope:SCOPE_DEV
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b64 v15, v16, s[12:13] scale_offset th:TH_LOAD_BYPASS scope:SCOPE_SYS
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b128 v17, v18, s[14:15] offset:64
  s_wait_loadcnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_cluster_saddr_inplace_end:
.size test_cluster_saddr_inplace, .Ltest_cluster_saddr_inplace_end-test_cluster_saddr_inplace

.rodata
.p2align 8
.amdhsa_kernel test_cluster_saddr_inplace
  .amdhsa_next_free_vgpr 20
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cluster_saddr_inplace
      .symbol: test_cluster_saddr_inplace.kd
      .sgpr_count: 16
      .vgpr_count: 20
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
