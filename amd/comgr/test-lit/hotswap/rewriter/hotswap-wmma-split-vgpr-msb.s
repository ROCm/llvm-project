// Verify that a K-split whose upper source half crosses v255 changes the
// gfx1250 VGPR-MSB mode around that half and restores the incoming mode.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API: WMMA split: patched v_wmma_f32_16x16x128_fp8_fp8
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_wmma_vgpr_msb>:
// DISASM:      s_branch
// DISASM:      v_wmma_f32_16x16x64_fp8_fp8 v[162:169], v[250:257], v[114:121], 0
// DISASM-NEXT: s_set_vgpr_msb 1
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8 v[162:169], v[2:9]{{.*v\[258:265\].*}}, v[122:129], v[162:169]
// DISASM-NEXT: s_set_vgpr_msb 0x100
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_vgpr_msb
.p2align 8
.type test_wmma_vgpr_msb,@function
test_wmma_vgpr_msb:
  s_set_vgpr_msb 0
  v_wmma_f32_16x16x128_fp8_fp8 v[162:169], v[250:265], v[114:129], 0
  s_endpgm
.size test_wmma_vgpr_msb, .-test_wmma_vgpr_msb
