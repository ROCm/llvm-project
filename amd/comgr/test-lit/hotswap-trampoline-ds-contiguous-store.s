// COM: A contiguous b32 DS2 store is one b64 store. Fold it in place so a
// COM: branch-target site does not need a 16-byte set-PC window across the
// COM: following conditional branch.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: ds_2addr: folded contiguous b32 store to b64
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_contiguous_store>:
// DISASM:      s_branch
// DISASM:      ds_store_b64 v7, v[14:15]
// DISASM-NEXT: v_cmpx_lt_u16_e32 63, v0.l
// DISASM-NEXT: s_cbranch_execz
// DISASM-NOT:  ds_store_2addr_b32
// DISASM-NOT:  s_add_pc_i64

// DISASM-LABEL: <test_contiguous_store_b64>:
// DISASM:      s_branch
// DISASM:      ds_store_b128 v10, v[4:7]
// DISASM-NEXT: v_cmpx_lt_u16_e32 63, v8.l
// DISASM-NEXT: s_cbranch_execz
// DISASM-NOT:  ds_store_2addr_b64

// COM: A foldable store immediately before a required non-foldable store must
// COM: remain available for a merged 16-byte far source window.
// DISASM-LABEL: <test_contiguous_store_merge_guard>:
// DISASM:      s_branch
// DISASM:      s_branch
// DISASM-NOT:  ds_store_2addr_b64
// DISASM:      s_cbranch_scc0

// COM: The reverse order is equally important: a foldable store must remain a
// COM: trampoline when a preceding adjacent required site is already queued.
// DISASM-LABEL: <test_contiguous_store_reverse_merge_guard>:
// DISASM:      s_branch
// DISASM:      s_branch
// DISASM-NOT:  ds_store_2addr_b64
// DISASM:      s_cbranch_scc0

// COM: A non-foldable store followed by a foldable store may use an existing
// COM: local cave for the first replacement; the neighbor then folds in place.
// COM: Do not force both sites into the appended pool merely because the
// COM: current site itself cannot fold.
// DISASM-LABEL: <test_contiguous_store_local_cave>:
// DISASM:      s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_store_b128 v6, v[8:11]
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      ds_store_b64 v6, v[0:1] offset:16
// DISASM-NEXT: ds_store_b64 v6, v[36:37] offset:24
// DISASM-NEXT: s_branch
// DISASM-NOT:  ds_store_2addr_b64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_contiguous_store
.p2align 8
.type test_contiguous_store,@function
test_contiguous_store:
  s_branch .Lstore
  s_mov_b32 s0, s1
.Lstore:
  ds_store_2addr_b32 v7, v14, v15 offset0:0 offset1:1
  v_cmpx_lt_u16_e32 63, v0.l
  s_cbranch_execz .Ldone
  s_mov_b32 s2, s3
.Ldone:
  s_endpgm
  .rept 40000
    s_mov_b32 s20, s21
  .endr
.Ltest_contiguous_store_end:
.size test_contiguous_store, .Ltest_contiguous_store_end-test_contiguous_store

.globl test_contiguous_store_b64
.p2align 8
.type test_contiguous_store_b64,@function
test_contiguous_store_b64:
  s_branch .Lstore_b64
  s_mov_b32 s0, s1
.Lstore_b64:
  ds_store_2addr_b64 v10, v[4:5], v[6:7] offset0:0 offset1:1
  v_cmpx_lt_u16_e32 63, v8.l
  s_cbranch_execz .Ldone_b64
  s_mov_b32 s2, s3
.Ldone_b64:
  s_endpgm
.Ltest_contiguous_store_b64_end:
.size test_contiguous_store_b64, .Ltest_contiguous_store_b64_end-test_contiguous_store_b64

.globl test_contiguous_store_merge_guard
.p2align 8
.type test_contiguous_store_merge_guard,@function
test_contiguous_store_merge_guard:
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  s_cbranch_scc0 .Ldone_merge_guard
.Ldone_merge_guard:
  s_endpgm
  .rept 40000
    s_mov_b32 s20, s21
  .endr
.Ltest_contiguous_store_merge_guard_end:
.size test_contiguous_store_merge_guard, .Ltest_contiguous_store_merge_guard_end-test_contiguous_store_merge_guard

.globl test_contiguous_store_reverse_merge_guard
.p2align 8
.type test_contiguous_store_reverse_merge_guard,@function
test_contiguous_store_reverse_merge_guard:
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  s_cbranch_scc0 .Ldone_reverse_merge_guard
.Ldone_reverse_merge_guard:
  s_endpgm
.Ltest_contiguous_store_reverse_merge_guard_end:
.size test_contiguous_store_reverse_merge_guard, .Ltest_contiguous_store_reverse_merge_guard_end-test_contiguous_store_reverse_merge_guard

.globl test_contiguous_store_local_cave
.p2align 8
.type test_contiguous_store_local_cave,@function
test_contiguous_store_local_cave:
  ds_store_2addr_b64 v6, v[0:1], v[36:37] offset0:2 offset1:3
  ds_store_2addr_b64 v6, v[8:9], v[10:11] offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
  .rept 16
    s_nop 0
  .endr
.Ltest_contiguous_store_local_cave_end:
.size test_contiguous_store_local_cave, .Ltest_contiguous_store_local_cave_end-test_contiguous_store_local_cave
