// COM: Complement to -cfg-live-scc: the SCC written by the mask-set is dead on
// COM: every path to the tensor because each successor of the join redefines
// COM: SCC (s_cmp) before any s_cbranch reads it. The forward liveness walk
// COM: clears the SCC bit on both join paths, so the definition-time clear is
// COM: safe and applies rather than deferring to the at-site fallback.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_cfg_scc_dead>:
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM-NOT: s_pack_hh_b32_b16

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_cfg_scc_dead
.p2align 8
.type test_tensor_cfg_scc_dead,@function
test_tensor_cfg_scc_dead:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  s_cbranch_execz .Lother
  s_cmp_eq_u32 s0, s0
  s_branch .Ltensor
.Lother:
  s_cmp_eq_u32 s1, s1
.Ltensor:
  s_cbranch_scc1 .Lend
.Lend:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_cfg_scc_dead_end:
.size test_tensor_cfg_scc_dead, .Ltest_tensor_cfg_scc_dead_end-test_tensor_cfg_scc_dead

.rodata
.p2align 8
.amdhsa_kernel test_tensor_cfg_scc_dead
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_cfg_scc_dead
      .symbol: test_tensor_cfg_scc_dead.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
