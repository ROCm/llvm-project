// COM: Verify that production entry displacement opts into whole-object
// COM: relocation when .dynamic follows .text. This layout requires moving
// COM: trailing section and program-header addresses.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// COM: Confirm .dynamic exists after .text in the input ELF.
// RUN: %llvm-readelf --section-headers %t.elf | %FileCheck --check-prefix=LAYOUT %s
// LAYOUT: .text
// LAYOUT: .dynamic

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --dump %t.out.elf --check-idempotent 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: displacement: grew ELF
// API-NOT: using appended entry stubs
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_elf_growth>:
// DISASM-NEXT: global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
// DISASM-NEXT: v_nop
// DISASM-NEXT: v_mov_b32_e32 v0, 0
// DISASM-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_elf_growth
.p2align 8
.type test_elf_growth,@function
test_elf_growth:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Ltest_elf_growth_end:
.size test_elf_growth, .Ltest_elf_growth_end-test_elf_growth

.rodata
.p2align 8
.amdhsa_kernel test_elf_growth
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
