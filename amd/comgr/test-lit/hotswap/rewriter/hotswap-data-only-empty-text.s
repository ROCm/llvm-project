// COM: Data-only HIP device objects can carry global constants and variables
// COM: but no kernels or device functions. Such an object has a present,
// COM: zero-size .text section and an empty amdhsa.kernels array. There are no
// COM: instructions or kernel revision tags to transform, so return a
// COM: byte-identical successful output after validating that shape.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
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
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.function.s -o %t.function.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.function.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=FUNCTION,REJECT %s
// FUNCTION: hotswap: error: data-only object has a function/ifunc symbol
// FUNCTION-SAME: in empty .text.

// RUN: sed 's/^\.set claimed_other_function, 0$/.set claimed_other_function, 1/' \
// RUN:   %s > %t.other-function.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
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
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.executable.s -o %t.executable.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.executable.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=EXECUTABLE,REJECT %s
// EXECUTABLE: hotswap: error: data-only object has non-empty executable
// EXECUTABLE-SAME: section '.other_text'.

// RUN: sed 's/^\.set claimed_descriptor, 0$/.set claimed_descriptor, 1/' \
// RUN:   %s > %t.descriptor.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.descriptor.s -o %t.descriptor.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.descriptor.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=DESCRIPTOR,REJECT %s
// DESCRIPTOR: hotswap: error: data-only object has kernel descriptor symbol
// DESCRIPTOR-SAME: 'claimed_kernel.kd'.

// RUN: sed 's/^\.set claimed_kernel, 0$/.set claimed_kernel, 1/' \
// RUN:   %s > %t.kernel.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.kernel.s -o %t.kernel.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.kernel.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=KERNEL,REJECT %s
// KERNEL: hotswap: error: data-only AMDGPU metadata claims 1 kernel(s).

// RUN: sed 's/^\.set malformed_metadata, 0$/.set malformed_metadata, 1/' \
// RUN:   %s > %t.malformed.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.malformed.s -o %t.malformed.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.malformed.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=MALFORMED,REJECT %s
// MALFORMED: hotswap: error: data-only validation:
// MALFORMED-SAME: failed to parse AMDGPU metadata note.

// RUN: %llvm-objcopy --remove-section=.text %t.elf %t.missing-text.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.missing-text.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefix=MISSING %s
// MISSING: no .text section found
// MISSING: RESULT: INVALID_ARGUMENT

// COM: A non-AMDGPU input must be rejected before any AMDGPU-specific
// COM: reasoning. An x86_64 relocatable from a single data definition has an
// COM: empty .text, one object symbol, and no notes; without an e_machine gate
// COM: it would be accepted for a gfx1250 rewrite.
// RUN: %yaml2obj %S/hotswap-data-only-foreign-machine.yaml -o %t.foreign.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.foreign.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=FOREIGN,REJECT %s
// FOREIGN: hotswap: error: data-only validation requires an AMDGPU ELF
// FOREIGN-SAME: (e_machine != EM_AMDGPU).

// COM: A section whose file range escapes the input buffer must be rejected
// COM: rather than copied out byte-for-byte. Here .rodata sh_offset is placed
// COM: far past the end of the object.
// RUN: %yaml2obj %S/hotswap-data-only-section-oob.yaml -o %t.section-oob.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.section-oob.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status INVALID_ARGUMENT 2>&1 \
// RUN:   | %FileCheck --check-prefixes=SECTION-OOB,REJECT %s
// SECTION-OOB: hotswap: error: data-only object has a section whose file range
// SECTION-OOB-SAME: lies outside the input buffer

// REJECT: hotswap: error: retargetCodeObject:
// REJECT-SAME: does not describe a valid data-only code object.
// REJECT: RESULT: INVALID_ARGUMENT

// COM: A relocatable object carries its AMDGPU metadata note only in the
// COM: section table (no PT_NOTE). This exercises the SHT_NOTE fallback that
// COM: every linked -nostdlib variant above leaves dead. The note has an empty
// COM: amdhsa.kernels array, so the object is data-only and accepted verbatim.
// RUN: %yaml2obj %S/hotswap-data-only-note-section.yaml -o %t.note-section.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.note-section.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.note-section.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=NOTE-SECTION %s
// NOTE-SECTION: hotswap: accepted data-only code object with empty .text;
// NOTE-SECTION-SAME: returning a byte-identical copy.
// NOTE-SECTION: RESULT: SUCCESS
// RUN: cmp %t.note-section.elf %t.note-section.out.elf

// COM: Boundary note for the executable_section case above: the byte-identical
// COM: data-only path rejects a non-empty executable section even when it is
// COM: not named .text (see .other_text). This documents that the empty-.text
// COM: acceptance never extends to objects that still carry executable code;
// COM: descriptorless callable libraries therefore take the normal rewrite
// COM: path, not this no-op copy.

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
