// A mandatory K=128 WMMA split in a block proven unreachable by a
// materialized-PC jump (s_get_pc_i64 / s_add_nc_u64 / s_set_pc_i64) out of the
// function must succeed using the ABI entry VGPR-MSB mode, not fail closed:
// the code never executes, so its incoming mode is semantically unobservable,
// but the A0-incompatible opcode must still be legalized. The first WMMA
// follows an explicit dead MODE write; the second has no local MODE write.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: error:
// API-COUNT-2: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API: applied 2 instruction patches
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// The A0-incompatible K=128 form must not survive; both halves are K=64. These
// operands stay in bank 0, so no s_set_vgpr_msb bracketing is required.
// DISASM-NOT: v_wmma_f32_16x16x128_fp8_fp8
// DISASM: v_wmma_f32_16x16x64_fp8_fp8

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_unreachable_cfg
.p2align 8
.type test_wmma_unreachable_cfg,@function
test_wmma_unreachable_cfg:
  // Generated get-PC/add/set-PC with a statically known target outside the
  // function, so the instructions below are CFG-unreachable.
  s_get_pc_i64 s[0:1]
  s_add_nc_u64 s[0:1], s[0:1], 0x1000
  s_set_pc_i64 s[0:1]

  s_set_vgpr_msb 0
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_delay_alu instid0(VALU_DEP_1)
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.Ltest_wmma_unreachable_cfg_end:
.size test_wmma_unreachable_cfg, .Ltest_wmma_unreachable_cfg_end-test_wmma_unreachable_cfg
