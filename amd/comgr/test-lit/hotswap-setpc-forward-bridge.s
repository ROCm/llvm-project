// COM: Once a far source window fits, bridge a short run of safe instructions
// COM: to merge the next required site rather than stranding it before a
// COM: control-flow instruction. No s_add_pc_i64 may be synthesized on A0.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: set-PC forward windows: expanded 1 far site(s), merged 1 adjacent trampoline site(s), synthesized zero s_add_pc_i64
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_forward_bridge>:
// DISASM:      s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-NOT:  ds_store_2addr_b64
// DISASM:      s_branch
// DISASM:      ds_store_b64 v2, v[0:1]
// DISASM-NEXT: ds_store_b64 v2, v[4:5] offset:8
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      v_mov_b32_e32 v0, v28
// DISASM-NEXT: ds_store_b64 v2, v[0:1] offset:16
// DISASM-NEXT: ds_store_b64 v2, v[6:7] offset:24
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      s_set_pc_i64
// DISASM-NOT:  s_add_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_forward_bridge
.p2align 8
.type test_forward_bridge,@function
test_forward_bridge:
  s_branch .Lfirst
.Lfirst:
  ds_store_2addr_b64 v2, v[0:1], v[4:5] offset0:0 offset1:1
  v_cndmask_b32_e64 v28, 0, 1.0, s2
  v_mov_b32_e32 v0, v28
  ds_store_2addr_b64 v2, v[0:1], v[6:7] offset0:2 offset1:3
  s_branch .Ldone
.Ldone:
  s_endpgm
  .rept 40000
    s_mov_b32 s20, s21
  .endr
.Ltest_forward_bridge_end:
.size test_forward_bridge, .Ltest_forward_bridge_end-test_forward_bridge

.rodata
.p2align 8
.amdhsa_kernel test_forward_bridge
  .amdhsa_next_free_vgpr 29
  .amdhsa_next_free_sgpr 96
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_forward_bridge
      .symbol: test_forward_bridge.kd
      .sgpr_count: 96
      .vgpr_count: 29
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
