// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > %t.elf
// RUN: hotswap-rewrite | %FileCheck --check-prefix=NULL %s
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx942 amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=INVALID %s
// RUN: hotswap-rewrite %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// RUN: hotswap-rewrite %t.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   | %FileCheck --check-prefix=MALFORMED %s

// NULL: NULL_ARGS: INVALID_ARGUMENT
// INVALID: RESULT: INVALID_ARGUMENT
// BADISA: RESULT: INVALID_ARGUMENT
// ZEROSIZE: RESULT: INVALID_ARGUMENT
// MALFORMED: RESULT: INVALID_ARGUMENT
