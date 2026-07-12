// COM: A required far DS rewrite may grow backward across the canonical
// COM: VALU_DEP_1 delay. The set-PC edge adds no VALU, so relocating the delay
// COM: together with its dependent instruction preserves which VALU it names.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: set-PC forward site {{.*}} expanded backward
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_backward_valu_delay>:
// DISASM:      s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-NEXT: s_branch
// DISASM-NOT:  ds_store_2addr_b64
// DISASM-NOT:  s_add_pc_i64
// DISASM:      s_delay_alu instid0(VALU_DEP_1)
// DISASM-NEXT: v_mov_b32_e32 v4, v13
// DISASM-NEXT: ds_store_b64 v2, v[0:1]
// DISASM-NEXT: ds_store_b64 v2, v[4:5] offset:8
// DISASM-NEXT: s_wait_dscnt 0x0

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_backward_valu_delay
.p2align 8
.type test_backward_valu_delay,@function
test_backward_valu_delay:
  v_cndmask_b32_e64 v13, 0, 1.0, s2
  s_delay_alu instid0(VALU_DEP_1)
  v_mov_b32_e32 v4, v13
  ds_store_2addr_b64 v2, v[0:1], v[4:5] offset1:1
  s_branch .Ltarget
.Ltarget:
  .rept 40000
    s_mov_b32 s20, s21
  .endr
  s_endpgm
.Ltest_backward_valu_delay_end:
.size test_backward_valu_delay, .Ltest_backward_valu_delay_end-test_backward_valu_delay

.rodata
.p2align 8
.amdhsa_kernel test_backward_valu_delay
  .amdhsa_next_free_vgpr 14
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_backward_valu_delay
      .symbol: test_backward_valu_delay.kd
      .sgpr_count: 24
      .vgpr_count: 14
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
