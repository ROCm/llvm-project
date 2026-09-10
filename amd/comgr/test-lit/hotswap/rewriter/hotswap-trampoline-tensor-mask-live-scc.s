// COM: The mask-set s_and defines SCC. When SCC is read before it is
// COM: redefined, changing the mask immediate could flip the branch condition,
// COM: so the definition clear is not applicable and the pass falls back to the
// COM: at-site rewrite, which does not touch the construction s_and.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: s4 dead, no save/restore needed
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// COM: The construction s_and keeps its original 0xfff7ffff immediate (the SCC
// COM: producer is untouched); the mask is cleared at the tensor instead.
// DISASM-LABEL: <test_tensor_live_scc>:
// DISASM: s_and_b32 s4, s4, 0xfff7ffff
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_live_scc
.p2align 8
.type test_tensor_live_scc,@function
test_tensor_live_scc:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  s_cbranch_scc1 .Lskip
  tensor_load_to_lds s[0:3], s[4:11]
.Lskip:
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_live_scc_end:
.size test_tensor_live_scc, .Ltest_tensor_live_scc_end-test_tensor_live_scc

.rodata
.p2align 8
.amdhsa_kernel test_tensor_live_scc
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_live_scc
      .symbol: test_tensor_live_scc.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
