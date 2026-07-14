// COM: A combined-delay DS2 prefix needs a contiguous relocation body, but the
// COM: only sufficiently large owned NOP sled is just outside direct source
// COM: reach. Relay both edges through a source-reachable, branch-only padding
// COM: island and retain the final delay/consumer suffix at its source address.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: hotswap: error:
// API: hotswap: DS2 gateway plan: used 1 entry gateway(s), 1 return gateway(s), and body sled at 0x20040 for site 0x0
// API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x0
// API-NOT: hotswap: error:
// API: hotswap: applied 1 instruction patches
// API: RESULT: SUCCESS

// RUN: %clang -x assembler-with-cpp -DSHORT_BODY -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.short.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.short.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=SHORT %s
// SHORT: hotswap: error: safe far return: no aligned block of 2 safe SGPRs fits below s106
// SHORT: RESULT: ERROR

// RUN: %clang -x assembler-with-cpp -DSECOND_DS2 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.second.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.second.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.second.out.elf 2>&1 | \
// RUN:   %FileCheck --check-prefix=SECOND-API %s
// SECOND-API-NOT: hotswap: error:
// SECOND-API: hotswap: DS2 gateway plan: used 1 entry gateway(s), 1 return gateway(s), and body sled at 0x20044 for site 0x0
// SECOND-API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x0 around protected site 0x8 with 2 DS2 rewrite(s)
// SECOND-API: hotswap: applied 2 instruction patches
// SECOND-API-NOT: hotswap: error:
// SECOND-API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.second.out.elf | \
// RUN:   %FileCheck --check-prefix=SECOND-DISASM %s
// SECOND-DISASM-LABEL: <combined_delay_gateway_relay>:
// SECOND-DISASM-NOT: ds_load_2addr_b64
// SECOND-DISASM: ds_load_b64 v[2:3], v52 offset:1224
// SECOND-DISASM-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// SECOND-DISASM-NEXT: s_wait_dscnt 0x0
// SECOND-DISASM-NEXT: ds_load_b64 v[6:7], v52 offset:1352
// SECOND-DISASM-NEXT: ds_load_b64 v[8:9], v52 offset:1416
// SECOND-DISASM-NEXT: s_wait_dscnt 0x0
// SECOND-DISASM-NOT: ds_load_2addr_b64
// RUN: hotswap-rewrite %t.second.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <combined_delay_gateway_relay>:
// DISASM-NEXT: s_branch {{.*}}<combined_delay_gateway_relay+0x10004>
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: v_cndmask_b32_e64 v116, 0, v198, s12
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch {{[0-9]+}}{{.*}}<combined_delay_gateway_relay+0x18>
// DISASM-NEXT: s_branch {{.*}}<combined_delay_gateway_relay+0x20040>
// DISASM: s_delay_alu instid0(SALU_CYCLE_1)
// DISASM-NEXT: s_or_b32 exec_lo, exec_lo, s12
// DISASM-NEXT: ds_load_b64 v[2:3], v52 offset:1224
// DISASM-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x1
// DISASM-NEXT: v_cmp_eq_u32_e64 s12, 0, v6
// DISASM-NEXT: s_branch {{.*}}<combined_delay_gateway_relay+0x10000>
// DISASM-NOT: ds_load_2addr_b64
// DISASM: <metadata_anchor>:

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local combined_delay_gateway_relay
.type combined_delay_gateway_relay,@function
combined_delay_gateway_relay:
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
#ifdef SECOND_DS2
  ds_load_2addr_b64 v[6:9], v52 offset0:169 offset1:177
#else
  s_wait_dscnt 1
#endif
  v_cmp_eq_u32_e64 s12, 0, v6
  v_cndmask_b32_e64 v116, 0, v198, s12

// The branch makes this zero-filled island unreachable by fallthrough. It is
// eligible for gateways only, not for the replacement body.
.rept (0x10000 - 0x24 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_branch .Lgateway_resume
.Lgateway_padding:
  .zero 8
.Lgateway_resume:

// Keep the body beyond direct source reach but within one short branch of the
// gateway. The original window needs 11 NOP dwords; two DS2 rewrites need 15.
.rept (0x20040 - 0x10008 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_set_pc_i64 s[30:31]
.Lbody_sled:
#ifdef SHORT_BODY
.rept 10
#elif defined(SECOND_DS2)
.rept 15
#else
.rept 11
#endif
  s_nop 0
.endr
.size combined_delay_gateway_relay, .-combined_delay_gateway_relay

.globl metadata_anchor
.type metadata_anchor,@function
metadata_anchor:
  s_endpgm
.size metadata_anchor, .-metadata_anchor

.rodata
.p2align 8
.amdhsa_kernel metadata_anchor
  .amdhsa_next_free_vgpr 199
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: metadata_anchor
      .symbol: metadata_anchor.kd
      .sgpr_count: 106
      .vgpr_count: 199
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
