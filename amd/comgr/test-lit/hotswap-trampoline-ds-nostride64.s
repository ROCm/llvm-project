// COM: Test HotSwap trampoline patch for the non-stride64 DS 2-address
// COM: family: ds_*_2addr_b{32,64} and ds_storexchg_2addr_rtn_b{32,64}.
// COM: Covers (1) b32 load, (2) b64 load, (3) b32 store, (4) b32 exchange,
// COM: (5) b64 store, (6) b64 exchange, and (7) a load+store+xchg
// COM: combination kernel that exercises the per-instruction dispatcher
// COM: in applyTrampolinePatchesImpl across multiple variant types in a
// COM: single trampoline pass.
// COM:
// COM: These differ from the stride64 forms in the byte-offset scale applied
// COM: to each per-operand index (ElemBytes vs 64 * ElemBytes). Every
// COM: non-stride family is lowered through a trampoline to two single-address
// COM: operations followed by an explicit DS drain. This includes cases whose
// COM: scaled offsets would fit the original DS2 fields: re-encoding only those
// COM: fields is not equivalent on hardware.
// COM:
// COM: Companion tests:
// COM:   hotswap-trampoline-ds-nostride64-multi.s -- drain insertion
// COM:     under multi-DS stacking in the non-stride64 path.
// COM:   hotswap-trampoline-ds-pipelined.s -- non-drain downstream wait
// COM:     preserved while each split adds its own drain.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// ---- Kernel 1: ds_load_2addr_b32 (non-stride64, byte offset = idx*4) -------
// COM: Kernel 1 (b32 load, non-stride64): offsets index*4. Source
// COM: offset0:4 offset1:8 -> byte offsets 16 and 32.
// DISASM-LABEL: <test_ds_load_b32_nostride64>:
// DISASM: ds_load_b32 v0, v2 offset:16
// DISASM-NEXT: ds_load_b32 v1, v2 offset:32
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: format as v[X:Y] register pairs.
// DISASM-LABEL: <test_ds_load_b64_nostride64>:
// DISASM: ds_load_b64 v[0:1], v4 offset:8
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:16
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: offsets 4 and 8.
// DISASM-LABEL: <test_ds_store_b32_nostride64>:
// DISASM: ds_store_b32 v2, v0 offset:4
// DISASM-NEXT: ds_store_b32 v2, v1 offset:8
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: offsets 4 and 12.
// DISASM-LABEL: <test_ds_xchg_b32_nostride64>:
// DISASM: ds_storexchg_rtn_b32 v0, v2, v3 offset:4
// DISASM-NEXT: ds_storexchg_rtn_b32 v1, v2, v4 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: Kernel 5 (b64 store, non-stride64): byte offsets 8 and 16 still fit the
// COM: DS2 fields, but hardware correctness still requires the split path.
// DISASM-LABEL: <test_ds_store_b64_nostride64>:
// DISASM: ds_store_b64 v4, v[0:1] offset:8
// DISASM-NEXT: ds_store_b64 v4, v[2:3] offset:16
// DISASM-NEXT: s_wait_dscnt 0x0

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

// ---- Kernel 5b: b64 scaled offset no longer fits the DS2 field -------------
// COM: Raw offsets 32 and 33 scale to 256 and 264. They fit the DS1 16-bit
// COM: field but not the DS2 8-bit fields. Verify that the common split path
// COM: also preserves these larger offsets.
// DISASM-LABEL: <test_ds_store_b64_nostride64_fallback>:
// DISASM-NOT: ds_store_2addr_b64
// DISASM: s_branch
// DISASM: ds_store_b64 v4, v[0:1] offset:256
// DISASM-NEXT: ds_store_b64 v4, v[2:3] offset:264

.globl test_ds_store_b64_nostride64_fallback
.p2align 8
.type test_ds_store_b64_nostride64_fallback,@function
test_ds_store_b64_nostride64_fallback:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:32 offset1:33
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
.Ltest_ds_store_b64_nostride64_fallback_end:
.size test_ds_store_b64_nostride64_fallback, .Ltest_ds_store_b64_nostride64_fallback_end-test_ds_store_b64_nostride64_fallback

// ---- Kernel 6: ds_storexchg_2addr_rtn_b64 (non-stride64 b64 exchange) ------
// COM: Kernel 6 (b64 exchange, non-stride64): exchange operand layout
// COM: (dst_pair, addr, data0_pair, data1_pair). Source offset0:1
// COM: offset1:2 -> byte offsets 8 and 16. Both vdst halves AND the data
// COM: operands format as v[X:Y] register pairs.
// DISASM-LABEL: <test_ds_xchg_b64_nostride64>:
// DISASM: ds_storexchg_rtn_b64 v[0:1], v8, v[4:5] offset:8
// DISASM-NEXT: ds_storexchg_rtn_b64 v[2:3], v8, v[6:7] offset:16
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: routes each variant to the common split lowering without state
// COM: leakage across types. All offsets scale by ElemBytes=4.
// COM:   ds_load_2addr_b32  offset0:1 offset1:2 -> byte 4, 8
// COM:   ds_store_2addr_b32 offset0:3 offset1:4 -> byte 12, 16
// COM:   ds_storexchg_*_b32 offset0:5 offset1:6 -> byte 20, 24
// DISASM-LABEL: <test_ds_combo_nostride64>:
// DISASM: ds_load_b32 v0, v8 offset:4
// DISASM-NEXT: ds_load_b32 v1, v8 offset:8
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM: ds_store_b32 v8, v2 offset:12
// DISASM-NEXT: ds_store_b32 v8, v3 offset:16
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM: ds_storexchg_rtn_b32 v4, v8, v6 offset:20
// DISASM-NEXT: ds_storexchg_rtn_b32 v5, v8, v7 offset:24
// DISASM-NEXT: s_wait_dscnt 0x0

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
// COM: bytes. Feeding the output back with its A0 stepping as the source
// COM: disables B0-to-A0 patches, so the A0 split operations are not changed
// COM: or scaled a second time.
// COM: Legacy ISA names without a stepping deliberately declare their source
// COM: as B0 on every invocation. Use the explicit A0 source here so the
// COM: idempotence check remains independent of other B0 patch families.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

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

.amdhsa_kernel test_ds_store_b64_nostride64_fallback
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
