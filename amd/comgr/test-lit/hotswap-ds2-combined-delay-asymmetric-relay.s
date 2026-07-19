// COM: Entry reaches the replacement body directly at the positive s_branch
// COM: limit, while the larger return origin needs one independent gateway.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: hotswap: error:
// API: hotswap: DS2 gateway plan: used 0 entry gateway(s), 1 return gateway(s), and body sled at 0x20000 for site 0x0
// API: hotswap: ds_2addr: demerged combined s_delay_alu at 0x0
// API-NOT: hotswap: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <combined_delay_asymmetric>:
// DISASM-NEXT: s_branch {{.*}}<combined_delay_asymmetric+0x20000>
// DISASM: s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: v_cndmask_b32_e64 v116, 0, v198, s12
// DISASM: s_branch {{[0-9]+}}{{.*}}<combined_delay_asymmetric+0x18>
// DISASM: v_cmp_eq_u32_e64 s12, 0, v6
// DISASM-NEXT: s_branch {{.*}}<combined_delay_asymmetric+0x10000>

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local combined_delay_asymmetric
.type combined_delay_asymmetric,@function
combined_delay_asymmetric:
  s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
  s_or_b32 exec_lo, exec_lo, s12
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_wait_dscnt 1
  v_cmp_eq_u32_e64 s12, 0, v6
  v_cndmask_b32_e64 v116, 0, v198, s12

.rept (0x10000 - 0x24 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_branch .Lreturn_gateway_resume
.Lreturn_gateway:
  .zero 4
.Lreturn_gateway_resume:

.rept (0x20000 - 0x10004 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_set_pc_i64 s[30:31]
.Lbody_sled:
.rept 11
  s_nop 0
.endr
.size combined_delay_asymmetric, .-combined_delay_asymmetric

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
