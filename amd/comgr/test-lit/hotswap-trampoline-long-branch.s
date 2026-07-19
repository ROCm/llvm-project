// COM: A tensor_load_to_lds must remain at its linked PC on gfx1250 A0. A
// COM: large .rept filler (~160 KB) makes any appended pool unreachable by
// COM: s_branch, but does not affect the in-place canonical-delay rewrite.
// COM: The function intentionally has st_size == 0, matching Tensile objects.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_far>:
// DISASM-NEXT: s_mov_b64 vcc, -1
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_endpgm

// METADATA: .name:           test_far
// METADATA: .sgpr_count:     14

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  s_mov_b64 vcc, -1
  s_delay_alu instid0(SALU_CYCLE_1)
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  // ~160 KB of non-NOP filler demonstrates that pool reachability is
  // irrelevant to the tensor patch.
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_far_end:

.rodata
.p2align 8
.amdhsa_kernel test_far
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_far
      .symbol: test_far.kd
      .sgpr_count: 14
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
