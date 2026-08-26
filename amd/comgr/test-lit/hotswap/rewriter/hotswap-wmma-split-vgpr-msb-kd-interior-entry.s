// A kernel-descriptor entry that points into a function interior is a real ABI
// entry point even when no intra-object branch or call reaches it. The VGPR-MSB
// analysis must seed that declared entry so the interior block is analyzed from
// the entry mode and the local s_set_vgpr_msb before the WMMA is honored,
// instead of being treated as unreachable and read as mode 0. The function
// symbol start terminates immediately, so the interior KD-entry block is only
// reachable through the descriptor; its s_set_vgpr_msb selects bank 0x60, so a
// correct split must bracket the crossing half for 0x60 (0x6050 / 0x5060). A
// mode-0 misread would emit no bracket at all.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: error:
// API-COUNT-1: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// The A0-incompatible K=128 form must not survive, and the seeded 0x60 mode
// must produce the bank-crossing brackets around the second K=64 half.
// DISASM-NOT: v_wmma_f32_16x16x128_fp8_fp8
// DISASM: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_set_vgpr_msb 0x6050
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_set_vgpr_msb 0x5060

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_kd_interior
.p2align 8
.type test_kd_interior,@function
test_kd_interior:
  // The symbol-start path terminates, so nothing falls through into the
  // interior block below; it is reachable only via the kernel-descriptor entry.
  s_endpgm
.Lkd_entry:
  s_set_vgpr_msb 0x60
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size test_kd_interior, .-test_kd_interior

// Hand-built kernel descriptor whose kernel_code_entry_byte_offset (at offset
// 0x10) resolves to the interior .Lkd_entry block rather than the symbol start.
.rodata
.p2align 6
.type test_kd_interior.kd,@object
.size test_kd_interior.kd, 64
test_kd_interior.kd:
  .zero 16
  .quad .Lkd_entry - test_kd_interior.kd
  .zero 40
