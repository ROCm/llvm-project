// COM: A pass-one far rewrite may place an exact materialized call-tail gateway
// COM: in certified NOP padding immediately after a sized function. The
// COM: gateway has no function owner, but its unique direct-call ingress,
// COM: terminal predecessor, exact instruction shape, and executable-pool
// COM: destination make it as statically known as an ordinary direct transfer.
// COM: Recognizing
// COM: it on pass two preserves global control-flow completeness and the
// COM: independent DS2 alignment proof.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.nometa.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nometa.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.nometa.out.elf 2>&1 | %FileCheck --check-prefix=FIRST %s
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nometa.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.nometa.out2.elf 2>&1 | %FileCheck --check-prefix=SECOND %s
// RUN: cmp %t.nometa.out.elf %t.nometa.out2.elf

// COM: A non-entry symbol inside the sequence invalidates the certificate.
// RUN: llvm-objcopy --add-symbol=gateway_interior=.text:0x2c,local \
// RUN:   %t.nometa.out.elf %t.interior-symbol.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.interior-symbol.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=INTERIOR %s

// COM: An empty kernel array is also not a target-state certificate.
// RUN: %clang -x assembler-with-cpp -DEMPTY_METADATA -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.empty.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.empty.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.empty.out.elf 2>&1 | %FileCheck --check-prefix=FIRST %s
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.empty.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.empty.out2.elf 2>&1 | %FileCheck --check-prefix=SECOND %s
// RUN: cmp %t.empty.out.elf %t.empty.out2.elf

// FIRST: ds_2addr: preserved proven-aligned ds_load_2addr_b64 at 0xC
// FIRST: assigned 1 SCC-neutral forward gateway(s)
// FIRST: RESULT: SUCCESS

// SECOND-NOT: incomplete control-flow targets
// SECOND: recognized materialized PC transfer [0x28, 0x34) -> 0x{{[0-9A-F]+}}
// SECOND-NOT: incomplete control-flow targets
// SECOND: ds_2addr: preserved proven-aligned ds_load_2addr_b64 at 0xC
// SECOND: applied 0 instruction patches
// SECOND: RESULT: SUCCESS

// INTERIOR-NOT: recognized materialized PC transfer [0x28, 0x34)
// INTERIOR: incomplete control-flow targets disable NOP padding donation
// INTERIOR: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// This B0 DS2 is legal on A0 because v4 is proven 8-byte aligned. It is
// intentionally retained on pass one and must remain proven on pass two.
.local aligned_site
.type aligned_site,@function
aligned_site:
  v_add_nc_u32_e64 v4, 0x4000, 0
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_endpgm
.size aligned_site, .-aligned_site

// This unproven DS2 forces an appended far trampoline. The definition after
// the site proves s[2:3] dead at the resume point, avoiding kernel metadata.
.local gateway_source
.type gateway_source,@function
gateway_source:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2
  s_mov_b64 s[2:3], 0
  s_endpgm
.size gateway_source, .-gateway_source

// Exactly one 20-byte generated gateway reservation. Its 12-byte call tail is
// unowned .text padding immediately after a sized, no-fallthrough function.
.rept 5
  s_nop 0
.endr

// Keep the appended trampoline more than one s_branch hop from the source and
// leave no intermediate NOP islands.
.rept 100000
  s_mov_b32 s0, s1
.endr

#ifdef EMPTY_METADATA
.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels: []
.end_amdgpu_metadata
#endif
