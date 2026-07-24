// COM: Test public HotSwap API validation of ELF identity and .text semantics.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status SUCCESS | %FileCheck --check-prefix=VALID %s
// VALID: RESULT: SUCCESS

// COM: e_machine must identify AMDGPU, not another ELF64 architecture.
// RUN: cp %t.elf %t.x86.elf
// RUN: printf '\x3e\x00' | dd of=%t.x86.elf bs=1 seek=18 conv=notrunc 2>/dev/null
// RUN: hotswap-rewrite %t.x86.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=INVALID %s

// COM: ET_CORE is not a supported GPU code-object layout.
// RUN: cp %t.elf %t.core.elf
// RUN: printf '\x04\x00' | dd of=%t.core.elf bs=1 seek=16 conv=notrunc 2>/dev/null
// RUN: hotswap-rewrite %t.core.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=INVALID %s

// COM: .text must have file-backed contents.
// RUN: %llvm-objcopy --set-section-type=.text=8 %t.elf %t.nobits.elf
// RUN: hotswap-rewrite %t.nobits.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=INVALID %s

// COM: .text must be both allocatable and executable.
// RUN: %llvm-objcopy --set-section-flags=.text=code,contents \
// RUN:   %t.elf %t.nonalloc.elf
// RUN: hotswap-rewrite %t.nonalloc.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=INVALID %s
// RUN: %llvm-objcopy --set-section-flags=.text=alloc,contents \
// RUN:   %t.elf %t.nonexec.elf
// RUN: hotswap-rewrite %t.nonexec.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=INVALID %s
// INVALID: RESULT: INVALID_ARGUMENT

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl elf_validation
.p2align 8
.type elf_validation,@function
elf_validation:
  s_endpgm
.Lelf_validation_end:
.size elf_validation, .Lelf_validation_end-elf_validation

.rodata
.p2align 8
.amdhsa_kernel elf_validation
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
