// Source relocation must fail closed for an unresolved set-PC transfer. A
// tail set-PC is a return only when MC proves it or it uses the ABI link pair;
// likewise, the rewriter's get-PC/add/set-PC exit shape is recognized only
// when no direct edge enters its interior.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.tail-nonlink.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.tail-nonlink.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=TAIL %s
// TAIL: source relocation disabled for function at 0x{{[0-9A-F]+}} by s_set_pc_i64 at 0x
// TAIL: WMMA split: protected site at 0x{{[0-9A-F]+}} cannot be relocated with an unresolved indirect entry
// TAIL: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.pc-relative-entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.pc-relative-entry.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=ALT %s
// ALT: source relocation disabled for function at 0x{{[0-9A-F]+}} by s_set_pc_i64 at 0x
// ALT: WMMA split: protected site at 0x{{[0-9A-F]+}} cannot be relocated with an unresolved indirect entry
// ALT: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.pc-relative-exit.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.pc-relative-exit.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.pc-relative-exit.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=KNOWN %s
// KNOWN-COUNT-2: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8 at offset 0x{{[0-9A-F]+}} with preceding delay in source window
// KNOWN: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_indirect_entry_reject
.p2align 8
.type test_wmma_indirect_entry_reject,@function
test_wmma_indirect_entry_reject:
#if CASE == 1
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  // An arbitrary computed tail transfer is not an ABI return.
  s_set_pc_i64 s[0:1]
#elif CASE == 2
  s_branch .Ladd
  s_get_pc_i64 s[0:1]
.Ladd:
  s_add_nc_u64 s[0:1], s[0:1], 0x1000
  s_set_pc_i64 s[0:1]
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
#elif CASE == 3
  // The exact generated shape with a statically known exit and no interior
  // entry is not an indirect destination ambiguity.
  s_get_pc_i64 s[0:1]
  s_add_nc_u64 s[0:1], s[0:1], 0x1000
  s_set_pc_i64 s[0:1]
  // Mandatory rewrites still patch CFG-unreachable instructions. Cover both
  // an explicit dead MODE write and a second dead site with no local MODE.
  s_set_vgpr_msb 0
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
#else
  .error "CASE must select a test body"
#endif
.Ltest_wmma_indirect_entry_reject_end:
.size test_wmma_indirect_entry_reject, .Ltest_wmma_indirect_entry_reject_end-test_wmma_indirect_entry_reject
