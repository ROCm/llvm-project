// A descriptor mask placed before tensor_load_to_lds is insufficient when a
// direct or unresolved indirect edge can enter at the PC-sensitive tensor and
// bypass the mask. Exercise both the already-masked and canonical-delay paths.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.direct.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.direct.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=DIRECT %s
// DIRECT: tensor_load_to_lds at 0x{{[0-9A-F]+}} may be entered without executing its descriptor mask
// DIRECT: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.indirect.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.indirect.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=INDIRECT %s
// INDIRECT: tensor_load_to_lds at 0x{{[0-9A-F]+}} may be entered without executing its descriptor mask
// INDIRECT: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_entry_reject
.p2align 8
.type test_tensor_entry_reject,@function
test_tensor_entry_reject:
#if CASE == 1
  s_branch .Ltensor
  s_pack_hh_b32_b16 s4, 0, s4
#elif CASE == 2
  s_set_pc_i64 s[0:1]
  s_delay_alu instid0(SALU_CYCLE_1)
#else
  .error "CASE must select a test body"
#endif
.Ltensor:
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_tensor_entry_reject_end:
.size test_tensor_entry_reject, .Ltest_tensor_entry_reject_end-test_tensor_entry_reject

.rodata
.p2align 8
.amdhsa_kernel test_tensor_entry_reject
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 28
.end_amdhsa_kernel
