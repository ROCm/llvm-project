// COM: tensor_load_to_lds is PC-sensitive on gfx1250 A0 and cannot fall back
// COM: to a NOP sled or appended trampoline. Mutate the instruction immediately
// COM: before the tensor between an ordinary s_nop and the canonical scalar
// COM: delay. The former must reach the missing-delay diagnostic; the latter
// COM: must rewrite in place at the same tensor PC.

// RUN: %clang -x assembler-with-cpp -DCANONICAL_DELAY=0 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.bad.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.bad.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=BAD %s
// BAD-NOT: may be entered without executing its descriptor mask
// BAD: hotswap: error: tensor_load_to_lds at 0x4 is not preceded by the canonical scalar delay
// BAD: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCANONICAL_DELAY=1 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.good.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.good.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.good.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=GOOD %s
// GOOD: tensor_load_to_lds: in-place descriptor mask at 0x0; tensor remains at 0x4
// GOOD: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.good.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_tensor_no_delay>:
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_no_delay
.p2align 8
.type test_tensor_no_delay,@function
test_tensor_no_delay:
#if CANONICAL_DELAY
  s_delay_alu instid0(SALU_CYCLE_1)
#else
  s_nop 0
#endif
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_no_delay_end:
.size test_tensor_no_delay, .Ltest_tensor_no_delay_end-test_tensor_no_delay

.rodata
.p2align 8
.amdhsa_kernel test_tensor_no_delay
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_no_delay
      .symbol: test_tensor_no_delay.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
