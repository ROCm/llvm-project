// A combined delay can protect both a source-side v_readfirstlane and the
// PC-sensitive tensor load after a K=128 WMMA. Splitting only the WMMA would
// leave the original delay naming a source branch. Demerge the two dependency
// IDs instead: relocate the first delay target, split WMMA, and barrier, then
// execute the second single-target delay from the vacated barrier slot. The
// tensor stays at its linked address.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: WMMA co-exec validation: 0 hazards (1 WMMA instructions scanned)
// API: hotswap: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} by demerging combined delay in source window
// API: hotswap: tensor_load_to_lds: masked local descriptor definition at 0x
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_delay_demerge>:
// DISASM: s_branch
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: tensor_load_to_lds s[24:27], s[4:11]
// DISASM: v_readfirstlane_b32 s4, v40
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM: s_delay_alu instid0(VALU_DEP_2)
// DISASM-NEXT: v_readfirstlane_b32 s19, v3
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_barrier_wait 0xffff

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_delay_demerge
.p2align 8
.type test_wmma_delay_demerge,@function
test_wmma_delay_demerge:
  v_readfirstlane_b32 s4, v40
  s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
  v_readfirstlane_b32 s19, v3
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_barrier_wait 0xffff
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_wmma_delay_demerge_end:
.size test_wmma_delay_demerge, .Ltest_wmma_delay_demerge_end-test_wmma_delay_demerge

.rodata
.p2align 8
.amdhsa_kernel test_wmma_delay_demerge
  .amdhsa_next_free_vgpr 41
  .amdhsa_next_free_sgpr 20
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_delay_demerge
      .symbol: test_wmma_delay_demerge.kd
      .sgpr_count: 20
      .vgpr_count: 41
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
