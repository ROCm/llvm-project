// COM: A proven arbitrary-link call into another function's interior bypasses
// COM: that function's ABI entry and VGPR-MSB initialization. Do not use the
// COM: entry-only MODE proof to exempt an aligned DS2 at the interior target.
// COM: A call to the actual function entry remains eligible for the proof.

// RUN: %clang -x assembler-with-cpp -DTARGET_INTERIOR=0 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.entry.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.entry.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.entry.out.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=LOG %s
// RUN: %llvm-objdump -d %t.entry.out.elf | \
// RUN:   %FileCheck --check-prefix=ENTRY %s

// RUN: %clang -x assembler-with-cpp -DTARGET_INTERIOR=1 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.interior.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.interior.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.interior.out.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=LOG %s
// RUN: %llvm-objdump -d %t.interior.out.elf | \
// RUN:   %FileCheck --check-prefix=INTERIOR %s
// RUN: hotswap-rewrite %t.interior.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// LOG: hotswap: resolved PC-materialized call
// LOG-NOT: hotswap: unresolved call target
// LOG: RESULT: SUCCESS
// ENTRY-LABEL: <materialized_cross_entry_target>:
// ENTRY: ds_load_2addr_b64
// ENTRY-LABEL: <materialized_cross_entry_caller>:
// INTERIOR-LABEL: <materialized_cross_entry_target>:
// INTERIOR-NOT: ds_load_2addr_b64
// INTERIOR-LABEL: <materialized_cross_entry_caller>:
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.local materialized_cross_entry_target
.p2align 8
.type materialized_cross_entry_target,@function
materialized_cross_entry_target:
  s_set_vgpr_msb 0
.Lmaterialized_cross_entry_interior:
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_endpgm
  .rept 8
    s_nop 0
  .endr
.Lmaterialized_cross_entry_target_end:
.size materialized_cross_entry_target, .Lmaterialized_cross_entry_target_end-materialized_cross_entry_target

.local materialized_cross_entry_caller
.p2align 8
.type materialized_cross_entry_caller,@function
materialized_cross_entry_caller:
  s_get_pc_i64 s[2:3]
#if TARGET_INTERIOR
  // Captured PC is caller+4; the aligned callee's interior is target+4.
  s_add_nc_u64 s[2:3], s[2:3], -256
#else
  // Captured PC is caller+4; the aligned callee entry is target+0.
  s_add_nc_u64 s[2:3], s[2:3], -260
#endif
  s_swap_pc_i64 s[0:1], s[2:3]
  s_endpgm
.Lmaterialized_cross_entry_caller_end:
.size materialized_cross_entry_caller, .Lmaterialized_cross_entry_caller_end-materialized_cross_entry_caller
