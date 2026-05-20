// COM: Test HotSwap trampoline patch for the non-stride64 DS 2-address
// COM: family: ds_*_2addr_b{32,64} and ds_storexchg_2addr_rtn_b{32,64}.
// COM: Covers (1) b32 load, (2) b64 load, (3) b32 store, (4) b32 exchange,
// COM: (5) b64 store, (6) b64 exchange, and (7) a load+store+xchg
// COM: combination kernel that exercises the per-instruction dispatcher
// COM: in applyTrampolinePatchesImpl across multiple variant types in a
// COM: single trampoline pass.
// COM:
// COM: These differ from the stride64 forms only in the byte-offset scale
// COM: applied to each per-operand index (ElemBytes vs 64 * ElemBytes), so
// COM: the trampoline shape (s_branch forward + sled + s_branch back) and
// COM: the s_wait_dscnt handling are shared with the stride64 path. Each
// COM: kernel here uses s_wait_dscnt 0x0 (drain) as input and exercises
// COM: the drain-preservation rule: the imm must stay at 0x0 after the
// COM: split (bumping a drain to K would let K split halves escape past
// COM: the wait).
// COM:
// COM: Companion tests:
// COM:   hotswap-trampoline-ds-nostride64-multi.s -- drain preservation
// COM:     under multi-DS stacking in the non-stride64 path.
// COM:   hotswap-trampoline-ds-pipelined.s -- non-drain bump path (covers
// COM:     the Ctx.Decoded writeback for the 0x1 -> 0x2 / 0x3 case
// COM:     originally raised in the ROCm/llvm-project#2281 review).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// ---- Kernel 1: ds_load_2addr_b32 (non-stride64, byte offset = idx*4) -------
// COM: Kernel 1 (b32 load, non-stride64): offsets index*4. Source
// COM: offset0:4 offset1:8 -> byte offsets 16 and 32. The input wait is
// COM: s_wait_dscnt 0x0 (drain) and stays at 0x0 after the split.
// DISASM-LABEL: <test_ds_load_b32_nostride64>:
// DISASM-NOT: ds_load_2addr_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v0, v2 offset:16
// DISASM-NEXT: ds_load_b32 v1, v2 offset:32
// DISASM: s_branch

.globl test_ds_load_b32_nostride64
.p2align 8
.type test_ds_load_b32_nostride64,@function
test_ds_load_b32_nostride64:
  ds_load_2addr_b32 v[0:1], v2 offset0:4 offset1:8
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_load_b32_nostride64_end:
.size test_ds_load_b32_nostride64, .Ltest_ds_load_b32_nostride64_end-test_ds_load_b32_nostride64

// ---- Kernel 2: ds_load_2addr_b64 (non-stride64, byte offset = idx*8) -------
// COM: Kernel 2 (b64 load, non-stride64): offsets index*8. Source
// COM: offset0:1 offset1:2 -> byte offsets 8 and 16. b64 destinations
// COM: format as v[X:Y] register pairs; drain wait stays at 0x0.
// DISASM-LABEL: <test_ds_load_b64_nostride64>:
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_load_b64 v[0:1], v4 offset:8
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:16
// DISASM: s_branch

.globl test_ds_load_b64_nostride64
.p2align 8
.type test_ds_load_b64_nostride64,@function
test_ds_load_b64_nostride64:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_load_b64_nostride64_end:
.size test_ds_load_b64_nostride64, .Ltest_ds_load_b64_nostride64_end-test_ds_load_b64_nostride64

// ---- Kernel 3: ds_store_2addr_b32 (non-stride64 store operand layout) ------
// COM: Kernel 3 (b32 store, non-stride64): store operand layout
// COM: (addr, data0, data1). Source offset0:1 offset1:2 -> byte
// COM: offsets 4 and 8; drain wait stays at 0x0.
// DISASM-LABEL: <test_ds_store_b32_nostride64>:
// DISASM-NOT: ds_store_2addr_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_store_b32 v2, v0 offset:4
// DISASM-NEXT: ds_store_b32 v2, v1 offset:8
// DISASM: s_branch

.globl test_ds_store_b32_nostride64
.p2align 8
.type test_ds_store_b32_nostride64,@function
test_ds_store_b32_nostride64:
  ds_store_2addr_b32 v2, v0, v1 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_store_b32_nostride64_end:
.size test_ds_store_b32_nostride64, .Ltest_ds_store_b32_nostride64_end-test_ds_store_b32_nostride64

// ---- Kernel 4: ds_storexchg_2addr_rtn_b32 (non-stride64 exchange layout) ---
// COM: Kernel 4 (b32 exchange, non-stride64): exchange operand layout
// COM: (dst, addr, data0, data1). Source offset0:1 offset1:3 -> byte
// COM: offsets 4 and 12; drain wait stays at 0x0.
// DISASM-LABEL: <test_ds_xchg_b32_nostride64>:
// DISASM-NOT: ds_storexchg_2addr_rtn_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_storexchg_rtn_b32 v0, v2, v3 offset:4
// DISASM-NEXT: ds_storexchg_rtn_b32 v1, v2, v4 offset:12
// DISASM: s_branch

.globl test_ds_xchg_b32_nostride64
.p2align 8
.type test_ds_xchg_b32_nostride64,@function
test_ds_xchg_b32_nostride64:
  ds_storexchg_2addr_rtn_b32 v[0:1], v2, v3, v4 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_xchg_b32_nostride64_end:
.size test_ds_xchg_b32_nostride64, .Ltest_ds_xchg_b32_nostride64_end-test_ds_xchg_b32_nostride64

// ---- Kernel 5: ds_store_2addr_b64 (non-stride64 store, b64 data pairs) -----
// COM: Kernel 5 (b64 store, non-stride64): store operand layout (addr,
// COM: data0_pair, data1_pair). Source offset0:1 offset1:2 -> byte
// COM: offsets 8 and 16. b64 data operands format as v[X:Y] register
// COM: pairs (exercises fmtRegOperand on the data side, complementing
// COM: kernel 2 which exercises it on the destination side); drain
// COM: wait stays at 0x0.
// DISASM-LABEL: <test_ds_store_b64_nostride64>:
// DISASM-NOT: ds_store_2addr_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_store_b64 v4, v[0:1] offset:8
// DISASM-NEXT: ds_store_b64 v4, v[2:3] offset:16
// DISASM: s_branch

.globl test_ds_store_b64_nostride64
.p2align 8
.type test_ds_store_b64_nostride64,@function
test_ds_store_b64_nostride64:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_store_b64_nostride64_end:
.size test_ds_store_b64_nostride64, .Ltest_ds_store_b64_nostride64_end-test_ds_store_b64_nostride64

// ---- Kernel 6: ds_storexchg_2addr_rtn_b64 (non-stride64 b64 exchange) ------
// COM: Kernel 6 (b64 exchange, non-stride64): exchange operand layout
// COM: (dst_pair, addr, data0_pair, data1_pair). Source offset0:1
// COM: offset1:2 -> byte offsets 8 and 16. Both vdst halves AND the data
// COM: operands format as v[X:Y] register pairs (exercises splitDstPair
// COM: AND fmtRegOperand on b64 within the xchg dispatch path, complementing
// COM: kernel 4 which exercises the b32 xchg layout); drain wait stays
// COM: at 0x0.
// DISASM-LABEL: <test_ds_xchg_b64_nostride64>:
// DISASM-NOT: ds_storexchg_2addr_rtn_b64
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: ds_storexchg_rtn_b64 v[0:1], v8, v[4:5] offset:8
// DISASM-NEXT: ds_storexchg_rtn_b64 v[2:3], v8, v[6:7] offset:16
// DISASM: s_branch

.globl test_ds_xchg_b64_nostride64
.p2align 8
.type test_ds_xchg_b64_nostride64,@function
test_ds_xchg_b64_nostride64:
  ds_storexchg_2addr_rtn_b64 v[0:3], v8, v[4:5], v[6:7] offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_xchg_b64_nostride64_end:
.size test_ds_xchg_b64_nostride64, .Ltest_ds_xchg_b64_nostride64_end-test_ds_xchg_b64_nostride64

// ---- Kernel 7: combination (load + store + xchg in one kernel) -------------
// COM: Kernel 7 (combination, non-stride64): a single function body mixes
// COM: ds_load_2addr_b32, ds_store_2addr_b32, and ds_storexchg_2addr_rtn_b32
// COM: before a single drain s_wait_dscnt 0x0. Verifies that the per-
// COM: instruction dispatcher in applyTrampolinePatchesImpl correctly
// COM: routes each variant to its own expansion type without state leakage
// COM: across types in a single trampoline pass; drain wait stays at 0x0
// COM: across all three split sites. Sleds appear in source order: load,
// COM: store, xchg. All offsets scale by ElemBytes=4.
// COM:   ds_load_2addr_b32  offset0:1 offset1:2 -> byte 4, 8
// COM:   ds_store_2addr_b32 offset0:3 offset1:4 -> byte 12, 16
// COM:   ds_storexchg_*_b32 offset0:5 offset1:6 -> byte 20, 24
// DISASM-LABEL: <test_ds_combo_nostride64>:
// DISASM-NOT: ds_load_2addr_b32
// DISASM-NOT: ds_store_2addr_b32
// DISASM-NOT: ds_storexchg_2addr_rtn_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// COM: Sled 1: load expansion.
// DISASM: ds_load_b32 v0, v8 offset:4
// DISASM-NEXT: ds_load_b32 v1, v8 offset:8
// COM: Sled 2: store expansion.
// DISASM: ds_store_b32 v8, v2 offset:12
// DISASM-NEXT: ds_store_b32 v8, v3 offset:16
// COM: Sled 3: xchg expansion.
// DISASM: ds_storexchg_rtn_b32 v4, v8, v6 offset:20
// DISASM-NEXT: ds_storexchg_rtn_b32 v5, v8, v7 offset:24

.globl test_ds_combo_nostride64
.p2align 8
.type test_ds_combo_nostride64,@function
test_ds_combo_nostride64:
  ds_load_2addr_b32 v[0:1], v8 offset0:1 offset1:2
  ds_store_2addr_b32 v8, v2, v3 offset0:3 offset1:4
  ds_storexchg_2addr_rtn_b32 v[4:5], v8, v6, v7 offset0:5 offset1:6
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds_combo_nostride64_end:
.size test_ds_combo_nostride64, .Ltest_ds_combo_nostride64_end-test_ds_combo_nostride64

// COM: Idempotency: rewriting the output again should produce identical
// COM: bytes (no DS2 mnemonic remains, second pass is a no-op).
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.rodata
.p2align 8
.amdhsa_kernel test_ds_load_b32_nostride64
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_load_b64_nostride64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_store_b32_nostride64
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_xchg_b32_nostride64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_store_b64_nostride64
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_xchg_b64_nostride64
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_combo_nostride64
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
