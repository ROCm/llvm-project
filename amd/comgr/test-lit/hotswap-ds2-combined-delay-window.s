// COM: Splitting a DS2 between the two targets of a combined s_delay_alu
// COM: changes the encoded instskip geometry. Restore its packed dependencies
// COM: as two standalone delays, keeping the final delay and its consumer as
// COM: a contiguous suffix at the end of the original source window.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: hotswap: error:
// API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x
// API-NOT: hotswap: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <combined_delay_ds2>:
// DISASM-NEXT: s_branch

// A four-byte final consumer uses an eight-byte retained suffix instead of the
// twelve-byte suffix above. The same structural rule must accept both widths.
// RUN: %clang -x assembler-with-cpp -DSHORT_CONSUMER -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.short.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.short.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.short.out.elf 2>&1 | %FileCheck --check-prefix=SHORT-API %s
// SHORT-API-NOT: hotswap: error:
// SHORT-API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x
// SHORT-API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.short.out.elf | %FileCheck --check-prefix=SHORT-DISASM %s
// SHORT-DISASM-LABEL: <combined_delay_ds2>:
// SHORT-DISASM-NEXT: s_branch
// SHORT-DISASM: s_delay_alu instid0(VALU_DEP_1)
// SHORT-DISASM-NEXT: v_min_i32_e32 v116, v116, v6
// SHORT-DISASM-NOT: ds_load_2addr_b64
// SHORT-DISASM: s_delay_alu instid0(SALU_CYCLE_1)
// SHORT-DISASM-NEXT: s_or_b32 exec_lo, exec_lo, s12
// SHORT-DISASM-NEXT: ds_load_b64 v[2:3], v52 offset:1224
// SHORT-DISASM-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// SHORT-DISASM-NEXT: s_wait_dscnt 0x0
// SHORT-DISASM-NEXT: s_wait_dscnt 0x1
// SHORT-DISASM-NEXT: v_cmp_eq_u32_e64 s12, 0, v6
// SHORT-DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: v_cndmask_b32_e64 v116, 0, v198, s12
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_delay_alu instid0(SALU_CYCLE_1)
// DISASM-NEXT: s_or_b32 exec_lo, exec_lo, s12
// DISASM-NEXT: ds_load_b64 v[2:3], v52 offset:1224
// DISASM-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x1
// DISASM-NEXT: v_cmp_eq_u32_e64 s12, 0, v6
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// A single combined delay can protect more than one required DS2 rewrite.
// Relocate and rewrite the complete window atomically rather than allowing the
// first replacement to suppress the second one.
// RUN: %clang -x assembler-with-cpp -DMULTI_DS2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.multi.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.multi.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.multi.out.elf 2>&1 | %FileCheck --check-prefix=MULTI-API %s
// MULTI-API-NOT: hotswap: error:
// MULTI-API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x{{[0-9A-F]+}} around protected site 0x{{[0-9A-F]+}} with 2 DS2 rewrite(s)
// MULTI-API: hotswap: applied 2 instruction patches
// MULTI-API-NOT: hotswap: error:
// MULTI-API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.multi.out.elf | \
// RUN:   %FileCheck --check-prefix=MULTI-DISASM %s
// MULTI-DISASM-LABEL: <combined_delay_ds2>:
// MULTI-DISASM-NEXT: s_branch
// MULTI-DISASM-NOT: ds_load_2addr_b32
// MULTI-DISASM: s_delay_alu instid0(VALU_DEP_1)
// MULTI-DISASM-NEXT: v_add_nc_u32_e32 v12, v4, v11
// MULTI-DISASM: s_delay_alu instid0(VALU_DEP_1)
// MULTI-DISASM-NEXT: v_add_nc_u32_e32 v10, v2, v10
// MULTI-DISASM-NEXT: ds_load_b32 v2, v58 offset:52
// MULTI-DISASM-NEXT: ds_load_b32 v3, v58 offset:56
// MULTI-DISASM-NEXT: s_wait_dscnt 0x0
// MULTI-DISASM-NEXT: ds_load_b32 v4, v58 offset:60
// MULTI-DISASM-NEXT: ds_load_b32 v5, v58 offset:64
// MULTI-DISASM-NEXT: s_wait_dscnt 0x0
// MULTI-DISASM-NEXT: s_wait_dscnt 0x3
// MULTI-DISASM-NEXT: v_add_nc_u32_e32 v11, v3, v10
// MULTI-DISASM-NOT: ds_load_2addr_b32
// RUN: hotswap-rewrite %t.multi.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=MULTI-IDEM %s
// MULTI-IDEM: IDEMPOTENT: YES

// RUN: %clang -x assembler-with-cpp -DMULTI_DS2 -DMIXED_DS2 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.mixed.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.mixed.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.mixed.out.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=MULTI-API %s
// RUN: %llvm-objdump -d %t.mixed.out.elf | \
// RUN:   %FileCheck --check-prefix=MIXED-DISASM %s
// MIXED-DISASM-LABEL: <combined_delay_ds2>:
// MIXED-DISASM-NOT: ds_store_2addr_b32
// MIXED-DISASM-NOT: ds_load_2addr_b32
// MIXED-DISASM: ds_store_b32 v58, v2 offset:52
// MIXED-DISASM-NEXT: ds_store_b32 v58, v3 offset:56
// MIXED-DISASM-NEXT: s_wait_dscnt 0x0
// MIXED-DISASM-NEXT: ds_load_b32 v4, v58 offset:60
// MIXED-DISASM-NEXT: ds_load_b32 v5, v58 offset:64
// MIXED-DISASM-NEXT: s_wait_dscnt 0x0
// MIXED-DISASM-NOT: ds_store_2addr_b32
// MIXED-DISASM-NOT: ds_load_2addr_b32
// RUN: hotswap-rewrite %t.mixed.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=MULTI-IDEM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl combined_delay_ds2
.p2align 8
.type combined_delay_ds2,@function
combined_delay_ds2:
#ifdef MULTI_DS2
  s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
  v_add_nc_u32_e32 v10, v2, v10
#ifdef MIXED_DS2
  ds_store_2addr_b32 v58, v2, v3 offset0:13 offset1:14
#else
  ds_load_2addr_b32 v[2:3], v58 offset0:13 offset1:14
#endif
  ds_load_2addr_b32 v[4:5], v58 offset0:15 offset1:16
  s_wait_dscnt 3
  v_add_nc_u32_e32 v11, v3, v10
  v_add_nc_u32_e32 v12, v4, v11
#else
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_wait_dscnt 1
  v_cmp_eq_u32_e64 s12, 0, v6
#ifdef SHORT_CONSUMER
  v_min_i32_e32 v116, v116, v6
#else
  v_cndmask_b32_e64 v116, 0, v198, s12
#endif
#endif
  s_endpgm
  .rept 32
    s_nop 0
  .endr
.Lcombined_delay_ds2_end:
.size combined_delay_ds2, .Lcombined_delay_ds2_end-combined_delay_ds2

.rodata
.p2align 8
.amdhsa_kernel combined_delay_ds2
  .amdhsa_next_free_vgpr 199
  .amdhsa_next_free_sgpr 13
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: combined_delay_ds2
      .symbol: combined_delay_ds2.kd
      .sgpr_count: 13
      .vgpr_count: 199
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
