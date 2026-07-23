// COM: Test HotSwap cluster_load in-place conversion across addressing forms.
// COM: Both the saddr=off (64-bit vaddr) form and the SGPR-relative (_SADDR)
// COM: form are neutralized on A0 by swapping the opcode in place to the
// COM: matching global_load_*: global_load never reads M0.wg_mask, so the
// COM: multicast is removed with no trampoline, scratch SGPR, or M0
// COM: save/clear/restore sequence. The _SADDR form maps to global_load_*_saddr
// COM: (distinct MC opcode, identical operand layout), so its scalar saddr and
// COM: 32-bit vaddr are preserved rather than mis-encoded as a 64-bit vaddr.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The _SADDR site is converted in place to global_load_b32 with its
// COM: scalar saddr operand preserved, and the saddr=off site to the off-form
// COM: global_load_b32. No cluster_load survives, and no trampoline branch or
// COM: M0 wg_mask save/clear/restore (s_pack_hh_b32_b16) is emitted.
// DISASM-LABEL: <test_saddr_kernel>:
// DISASM-NOT: cluster_load
// DISASM: global_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NOT: cluster_load
// DISASM: global_load_b32 v{{[0-9]+}}, v[{{[0-9:]+}}], off
// DISASM-NOT: cluster_load
// DISASM: s_endpgm
// DISASM-NOT: cluster_load
// DISASM-NOT: s_pack_hh_b32_b16
// DISASM-NOT: s_branch

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_saddr_kernel
.p2align 8
.type test_saddr_kernel,@function
test_saddr_kernel:
  // SGPR-relative (SADDR) form -- converted to global_load_b32 v4, v1, s[2:3].
  cluster_load_b32 v4, v1, s[2:3]
  s_wait_loadcnt 0x0
  // saddr=off form -- converted to global_load_b32 v5, v[2:3], off.
  cluster_load_b32 v5, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_saddr_kernel_end:
.size test_saddr_kernel, .Ltest_saddr_kernel_end-test_saddr_kernel

.rodata
.p2align 8
.amdhsa_kernel test_saddr_kernel
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_saddr_kernel
      .symbol: test_saddr_kernel.kd
      .sgpr_count: 4
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
