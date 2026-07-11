// COM: HSV-009 / PLAT-205406: LLVM used to encode a modest negative
// COM: s_add_pc_i64 displacement with a 64-bit literal, while the equivalent
// COM: positive displacement used a 32-bit literal. The 64-bit-literal return
// COM: corrupts wave state on gfx1250 A0. MI400 defines the literal32 form as
// COM: sign-extended, so both trampoline edges can use the safe 8-byte form
// COM: without scratch SGPRs.
// COM: A large .rept filler (~160 KB, non-NOP so it forms no usable sled)
// COM: pushes the pool past s_branch's reach to force the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_far>:
// DISASM-NEXT: s_mov_b64 vcc, -1
// DISASM-NEXT: s_add_pc_i64
// DISASM-NOT: s_add_pc_i64
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_add_pc_i64 0xffff{{[0-9a-f]+}}

// METADATA: .name:           test_far
// METADATA: .sgpr_count:     14

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: A kernel using all 106 numbered SGPRs still patches: control flow needs
// COM: no register and metadata remains unchanged.
// RUN: sed -e 's/s_mov_b64 vcc, -1/s_mov_b32 s105, 0/' \
// RUN:   -e 's/\.amdhsa_next_free_sgpr 12/.amdhsa_next_free_sgpr 106/' \
// RUN:   -e 's/\.sgpr_count: 14/.sgpr_count: 106/' %s > %t.full-sgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.full-sgpr.s -o %t.full-sgpr.elf
// RUN: hotswap-rewrite %t.full-sgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.full-sgpr.out.elf | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.full-sgpr.out.elf \
// RUN:   | %FileCheck --check-prefix=FULL-SGPR %s
// FULL-SGPR-LABEL: <test_far>:
// FULL-SGPR: s_add_pc_i64
// FULL-SGPR: tensor_load_to_lds
// FULL-SGPR-NEXT: s_add_pc_i64 0xffff{{[0-9a-f]+}}

// COM: A metadata-less object also patches because no resource count changes.
// RUN: sed '/^.amdgpu_metadata$/,/^.end_amdgpu_metadata$/d' %s > %t.nometa.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.nometa.s -o %t.nometa.elf
// RUN: hotswap-rewrite %t.nometa.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.nometa.out.elf | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.nometa.out.elf \
// RUN:   | %FileCheck --check-prefix=FULL-SGPR %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  s_mov_b64 vcc, -1
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the tensor_load above (forces the
  // long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_far_end:
.size test_far, .Ltest_far_end-test_far

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
