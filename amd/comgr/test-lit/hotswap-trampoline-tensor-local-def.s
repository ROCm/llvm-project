// COM: A gfx1250 A0 tensor descriptor can be formed well before the tensor
// COM: load, leaving no disposable scalar delay immediately before the load.
// COM: When a straight-line v_readfirstlane uniquely defines the descriptor
// COM: base, relocate that definition with the multicast mask and leave the
// COM: PC-sensitive tensor instruction at its linked address. The intervening
// COM: K=128 WMMA and its single-instruction delay move as one source window,
// COM: so the delay still immediately precedes the split sequence rather than
// COM: incorrectly naming the source branch.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: WMMA co-exec validation: 0 hazards (1 WMMA instructions scanned)
// API: hotswap: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} with preceding delay in source window
// API: hotswap: tensor_load_to_lds: masked local descriptor definition at 0x
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_local_def>:
// DISASM: s_branch
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: tensor_load_to_lds s[24:27], s[4:11]
// DISASM: v_readfirstlane_b32 s4, v40
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM: s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_local_def
.p2align 8
.type test_tensor_local_def,@function
test_tensor_local_def:
  v_readfirstlane_b32 s4, v40
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_tensor_local_def_end:
.size test_tensor_local_def, .Ltest_tensor_local_def_end-test_tensor_local_def

.rodata
.p2align 8
.amdhsa_kernel test_tensor_local_def
  .amdhsa_next_free_vgpr 41
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_local_def
      .symbol: test_tensor_local_def.kd
      .sgpr_count: 12
      .vgpr_count: 28
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
