// COM: A0 tensor multicast-mask clearing with the descriptor built in two
// COM: mutually exclusive wave branches (odd/even), as cluster kernels do.
// COM: The backward reaching-definition scan covers both construction regions,
// COM: so the mask-set s_and of each is cleared.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API: hotswap: tensor_load_to_lds: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// COM: Both wave regions have their mask-set cleared; neither pre-mask normalize
// COM: is touched.
// DISASM-LABEL: <test_tensor_mask_wave>:
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM: s_and_b32 s4, s4, 0xfff70000
// DISASM: tensor_load_to_lds s[0:3], s[4:11]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_wave
.p2align 8
.type test_tensor_mask_wave,@function
test_tensor_mask_wave:
  s_cbranch_scc1 .Lodd
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfff7ffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
  s_branch .Ldone
.Lodd:
  s_mov_b32 s4, 0
  s_and_b32 s4, s4, 0xfff7ffff
  s_or_b32 s4, s4, s5
  s_and_b32 s4, s4, 0xfff7ffff
.Ldone:
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
.Ltest_tensor_mask_wave_end:
.size test_tensor_mask_wave, .Ltest_tensor_mask_wave_end-test_tensor_mask_wave

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_wave
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_wave
      .symbol: test_tensor_mask_wave.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
