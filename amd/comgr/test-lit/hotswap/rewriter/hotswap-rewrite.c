// clang-format off
// COM: Test HotSwap rewrite API

// COM: Create a minimal test ELF file (ELF64 header only, no sections).
// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > %t.elf

// COM: NULL-argument validation (no args)
// RUN: hotswap-rewrite | %FileCheck --check-prefix=NULL %s
// NULL: NULL_ARGS: INVALID_ARGUMENT

// COM: Options API validation
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --bad-options-size --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=BADOPTIONS %s
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --bad-options-flags --expect-status INVALID_ARGUMENT \
// RUN:   | %FileCheck --check-prefix=BADOPTIONS %s
// BADOPTIONS: RESULT: INVALID_ARGUMENT

// COM: Unsupported ISA pair
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx942 amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=INVALID %s
// INVALID: RESULT: INVALID_ARGUMENT

// COM: Invalid ISA string
// RUN: hotswap-rewrite %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// BADISA: RESULT: INVALID_ARGUMENT

// COM: Zero-size input with supported ISA
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// ZEROSIZE: RESULT: INVALID_ARGUMENT

// COM: Supported GFX1250 pair on a malformed ELF (no section table and no
// COM: .text section). retargetCodeObject rejects inputs that fail ELF64
// COM: parsing instead of silently returning an unchanged successful copy.
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   | %FileCheck --check-prefix=MALFORMED %s
// MALFORMED: RESULT: INVALID_ARGUMENT

// COM: End-to-end coverage on real gfx1250 code objects is covered by
// COM: hotswap-rewrite-e2e.hip and hotswap-kernel-entry-trampoline.s.
