// COM: A descriptorless object with no revision markers is still an
// COM: all-missing legacy input. Generic ISA defaults it to B0, so the
// COM: canonical split path is safe and byte-idempotent.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: generic gfx1250 source has no .gfx1250_revision markers
// LOG-SAME: canonical split form
// LOG-NOT: rewrote ds_store_2addr_b64
// API: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_ds2_descriptorless_no_marker>:
// DISASM-NOT: ds_store_2addr_b64
// DISASM: s_branch
// DISASM: ds_store_b64 v4, v[0:1]
// DISASM-NEXT: ds_store_b64 v4, v[2:3] offset:8
// DISASM-NEXT: s_wait_dscnt 0x0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_descriptorless_no_marker
.p2align 8
.type test_ds2_descriptorless_no_marker,@function
test_ds2_descriptorless_no_marker:
  ds_store_2addr_b64 v4, v[0:1], v[2:3] offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_ds2_descriptorless_no_marker_end:
.size test_ds2_descriptorless_no_marker, .Ltest_ds2_descriptorless_no_marker_end-test_ds2_descriptorless_no_marker
