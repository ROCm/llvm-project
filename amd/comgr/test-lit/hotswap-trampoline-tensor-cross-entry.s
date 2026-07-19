// The kernel-entry EXEC seed and descriptor facts are valid only for dispatch.
// Reject non-dispatch ingress from another original-.text function, including
// an interior branch, s_call_i64's operand-1 target, and materialized set-PC.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.branch-interior.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.branch-interior.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.call-entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.call-entry.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.setpc-interior.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.setpc-interior.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=4 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.branch-entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.branch-entry.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=5 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.indirect-call.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.indirect-call.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=6 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.unknown-owned.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.unknown-owned.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=7 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.fallthrough-entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.fallthrough-entry.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s

// NEG-NOT: descriptor low16 already zero
// NEG: hotswap: error: tensor_load_to_lds at 0x
// NEG: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl tensor_external_ingress
.type tensor_external_ingress,@function
tensor_external_ingress:
#if CASE == 1
  s_branch .Ltensor
#elif CASE == 2
  s_call_i64 s[30:31], test_tensor_cross_entry
#elif CASE == 3
.Lgetpc:
  s_get_pc_i64 s[0:1]
  s_add_nc_u64 s[0:1], s[0:1], .Ltensor-(.Lgetpc+4)
  s_set_pc_i64 s[0:1]
#elif CASE == 4
  s_branch test_tensor_cross_entry
#elif CASE == 5
  s_swap_pc_i64 s[30:31], s[0:1]
#elif CASE == 6
  .long 0xffffffff
#elif CASE == 7
  s_nop 0
#else
#error unsupported CASE
#endif
#if CASE != 7
  s_endpgm
#endif
.Ltensor_external_ingress_end:
.size tensor_external_ingress, .Ltensor_external_ingress_end-tensor_external_ingress

.globl test_tensor_cross_entry
#if CASE != 7
.p2align 8
#endif
.type test_tensor_cross_entry,@function
test_tensor_cross_entry:
  v_mov_b32 v0, 0
  v_readfirstlane_b32 s4, v0
  s_cmp_eq_u32 s4, 0
  s_nop 0
.Ltensor:
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_tensor_cross_entry_end:
.size test_tensor_cross_entry, .Ltest_tensor_cross_entry_end-test_tensor_cross_entry

.rodata
.p2align 8
.amdhsa_kernel test_tensor_cross_entry
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_cross_entry
      .symbol: test_tensor_cross_entry.kd
      .sgpr_count: 32
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
