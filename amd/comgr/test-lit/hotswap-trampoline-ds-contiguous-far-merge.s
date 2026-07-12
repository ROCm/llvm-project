// COM: Mixed foldable/non-foldable adjacent DS2 stores must remain together in
// COM: the deferred set. This covers both orders and provides no local cave, so
// COM: each pair must merge into one 16-byte far source window.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: set-PC forward windows: expanded 2 far site(s), merged 2 adjacent trampoline site(s), synthesized zero s_add_pc_i64
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_contiguous_far_merge>:
// DISASM-COUNT-2: s_get_pc_i64
// DISASM-NOT: ds_store_2addr_b64
// DISASM:      ds_store_b128
// DISASM-NEXT: ds_store_b64
// DISASM-NEXT: ds_store_b64
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      ds_store_b64
// DISASM-NEXT: ds_store_b64
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_store_b128
// DISASM-NOT: s_add_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_contiguous_far_merge
.p2align 8
.type test_contiguous_far_merge,@function
test_contiguous_far_merge:
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  s_cbranch_scc0 .Lreverse
  s_mov_b32 s20, s21
.Lreverse:
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  s_cbranch_scc0 .Lfiller
.Lfiller:
  .rept 40000
    s_mov_b32 s20, s21
  .endr
  s_endpgm
.Ltest_contiguous_far_merge_end:
.size test_contiguous_far_merge, .Ltest_contiguous_far_merge_end-test_contiguous_far_merge

.rodata
.p2align 8
.amdhsa_kernel test_contiguous_far_merge
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_contiguous_far_merge
      .symbol: test_contiguous_far_merge.kd
      .sgpr_count: 24
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
