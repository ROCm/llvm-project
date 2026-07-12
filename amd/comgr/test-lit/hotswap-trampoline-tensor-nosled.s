// COM: tensor_load_to_lds is PC-sensitive on gfx1250 A0 and cannot fall back
// COM: to a NOP sled or appended trampoline. If the compiler's canonical
// COM: scalar delay is absent, the required multicast fix must fail.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds at 0x0 has no preceding delay slot
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_trampoline
.p2align 8
.type test_tensor_trampoline,@function
test_tensor_trampoline:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_trampoline_end:
.size test_tensor_trampoline, .Ltest_tensor_trampoline_end-test_tensor_trampoline

.rodata
.p2align 8
.amdhsa_kernel test_tensor_trampoline
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
