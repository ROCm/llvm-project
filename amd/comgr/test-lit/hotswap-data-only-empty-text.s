// COM: Data-only HIP device objects can carry global constants and variables
// COM: but no kernels or device functions. Such an object has a present,
// COM: zero-size .text section and an empty amdhsa.kernels array. There are no
// COM: instructions or kernel revision tags to transform, so return a
// COM: byte-identical successful output after validating that shape.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: %llvm-readobj --sections --symbols --notes %t.elf \
// RUN:   | %FileCheck --check-prefix=SHAPE --implicit-check-not=.kd %s
// SHAPE: Name: .text
// SHAPE: Size: 0
// SHAPE: Name: data_only_constant
// SHAPE: Type: Object
// SHAPE: AMDGPU Metadata: ---
// SHAPE-NEXT: amdhsa.kernels:  []

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=ACCEPT %s
// ACCEPT: hotswap: accepted data-only code object with empty .text;
// ACCEPT-SAME: returning a byte-identical copy.
// ACCEPT: RESULT: SUCCESS
// RUN: cmp %t.elf %t.out.elf
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES
// RUN: env AMD_COMGR_TIME_STATISTICS=1 \
// RUN:   AMD_COMGR_TIME_STATISTICS_GRANULARITY=ns \
// RUN:   AMD_COMGR_REDIRECT_LOGS=%t.profile.log \
// RUN:   hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.profile.elf
// RUN: %FileCheck --check-prefix=PROFILE-PARSE \
// RUN:   --input-file=%t.profile.log %s
// RUN: %FileCheck --check-prefix=PROFILE-COPY \
// RUN:   --input-file=%t.profile.log %s
// PROFILE-PARSE: phase:elf_parse{{ +}}1 calls
// PROFILE-COPY: phase:output_copy{{ +}}1 calls

// RUN: sed 's/^\.set claimed_function, 0$/.set claimed_function, 1/' \
// RUN:   %s > %t.function.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.function.s -o %t.function.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.function.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=FUNCTION,REJECT %s
// FUNCTION: hotswap: error: data-only object has a function/ifunc symbol
// FUNCTION-SAME: in empty .text.

// RUN: sed 's/^\.set claimed_other_function, 0$/.set claimed_other_function, 1/' \
// RUN:   %s > %t.other-function.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.other-function.s -o %t.other-function.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   hotswap-rewrite %t.other-function.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=OTHER-FUNCTION,REJECT %s
// OTHER-FUNCTION: hotswap: error: data-only object has defined function/ifunc
// OTHER-FUNCTION-SAME: symbol 'claimed_other_function'.

// RUN: sed 's/^\.set executable_section, 0$/.set executable_section, 1/' \
// RUN:   %s > %t.executable.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.executable.s -o %t.executable.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.executable.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=EXECUTABLE,REJECT %s
// EXECUTABLE: hotswap: error: data-only object has non-empty executable
// EXECUTABLE-SAME: section '.other_text'.

// RUN: sed 's/^\.set claimed_descriptor, 0$/.set claimed_descriptor, 1/' \
// RUN:   %s > %t.descriptor.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.descriptor.s -o %t.descriptor.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.descriptor.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=DESCRIPTOR,REJECT %s
// DESCRIPTOR: hotswap: error: data-only object has kernel descriptor symbol
// DESCRIPTOR-SAME: 'claimed_kernel.kd'.

// RUN: sed 's/^\.set claimed_kernel, 0$/.set claimed_kernel, 1/' \
// RUN:   %s > %t.kernel.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.kernel.s -o %t.kernel.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.kernel.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=KERNEL,REJECT %s
// KERNEL: hotswap: error: data-only AMDGPU metadata claims 1 kernel(s).

// RUN: sed 's/^\.set malformed_metadata, 0$/.set malformed_metadata, 1/' \
// RUN:   %s > %t.malformed.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.malformed.s -o %t.malformed.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.malformed.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=MALFORMED,REJECT %s
// MALFORMED: hotswap: error: failed to parse data-only AMDGPU metadata note.

// RUN: %llvm-objcopy --remove-section=.text %t.elf %t.missing-text.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.missing-text.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefix=MISSING %s
// MISSING: no .text section found
// MISSING: RESULT: INVALID_ARGUMENT

// REJECT: hotswap: error: retargetCodeObject:
// REJECT-SAME: does not describe a valid data-only code object.
// REJECT: RESULT: INVALID_ARGUMENT

.set claimed_function, 0
.set claimed_other_function, 0
.set executable_section, 0
.set claimed_descriptor, 0
.set claimed_kernel, 0
.set malformed_metadata, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.if claimed_function
.globl claimed_function
.type claimed_function,@function
claimed_function:
.size claimed_function, .-claimed_function
.endif

.rodata
.globl data_only_constant
.type data_only_constant,@object
.p2align 2
data_only_constant:
  .long 1
.size data_only_constant, .-data_only_constant

.if claimed_other_function
.globl claimed_other_function
.type claimed_other_function,@function
claimed_other_function:
.size claimed_other_function, .-claimed_other_function
.endif

.if executable_section
.section .other_text,"ax",@progbits
  v_nop
.endif

.if claimed_descriptor
.globl claimed_kernel.kd
.type claimed_kernel.kd,@object
.p2align 6
claimed_kernel.kd:
  .zero 64
.size claimed_kernel.kd, .-claimed_kernel.kd
.endif

.if malformed_metadata
.section .note,"a",@note
.p2align 2
  .long 7
  .long 4
  .long 32
  .asciz "AMDGPU"
.p2align 2
  .byte 0xc1, 0xc1, 0xc1, 0xc1
.p2align 2
.else
.section .note,"a",@note
.p2align 2
  .long 7
  .long .Lmetadata_desc_end-.Lmetadata_desc_begin
  .long 32
  .asciz "AMDGPU"
.p2align 2
.Lmetadata_desc_begin:
.if claimed_kernel
  // {"amdhsa.kernels": [{}]}
  .byte 0x81, 0xae
  .ascii "amdhsa.kernels"
  .byte 0x91, 0x80
.else
  // Match the corpus note:
  // {"amdhsa.kernels": [], "amdhsa.target": "...gfx1250",
  //  "amdhsa.version": [1, 2]}
  .byte 0x83, 0xae
  .ascii "amdhsa.kernels"
  .byte 0x90, 0xad
  .ascii "amdhsa.target"
  .byte 0xba
  .ascii "amdgcn-amd-amdhsa--gfx1250"
  .byte 0xae
  .ascii "amdhsa.version"
  .byte 0x92, 0x01, 0x02
.endif
.Lmetadata_desc_end:
.p2align 2
.endif
