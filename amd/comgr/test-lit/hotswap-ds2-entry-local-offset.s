// COM: Reduced from corpus object
// COM: 8737ad6b480494d2a19aa82de8dfdd53e6c4b788975bd57e3c3437bc368ea780.
// COM: The original B0 ds_store_2addr_b64 uses element offsets (0, 1), which
// COM: A0 must interpret as byte offsets (0, 8). Both corrected values fit the
// COM: original DS2 fields, so patch the two immediate bytes in place.
// COM:
// COM: The unresolved register call deliberately keeps indirect entry points
// COM: unbounded, and the filler makes any appended pool unreachable by
// COM: s_branch. Success therefore proves that this rewrite is entry-local:
// COM: it neither consumes neighboring instructions nor weakens the unknown
// COM: control-flow guards.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: generic gfx1250 source has uniform B0 metadata
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: rewrote ds_store_2addr_b64 at 0x{{[0-9A-F]+}} in place with A0 byte offsets 0, 8
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s
// DISASM-LABEL: <test_ds2_entry_local>:
// DISASM-NEXT: s_swap_pc_i64 s[30:31], s[0:1]
// DISASM-NEXT: ds_store_2addr_b64 v19, v[14:15], v[20:21] offset1:8
// DISASM-NEXT: s_wait_loadcnt_dscnt 0x1
// DISASM-NEXT: s_endpgm
// METADATA: .gfx1250_revision: A0

// COM: Repeat the same generic source/target request. The first pass's uniform
// COM: A0 revision metadata is the durable proof that the surviving DS2
// COM: offsets are already byte offsets, so the second output is byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_entry_local
.p2align 8
.type test_ds2_entry_local,@function
test_ds2_entry_local:
  s_swap_pc_i64 s[30:31], s[0:1]
  ds_store_2addr_b64 v19, v[14:15], v[20:21] offset1:1
  s_wait_loadcnt_dscnt 0x1
  s_endpgm
.size test_ds2_entry_local, .-test_ds2_entry_local

// Keep a hypothetical appended trampoline pool outside signed s_branch reach.
.rept 40000
  s_mov_b32 s2, s3
.endr

.rodata
.p2align 8
.amdhsa_kernel test_ds2_entry_local
  .amdhsa_next_free_vgpr 22
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_ds2_entry_local
      .symbol: test_ds2_entry_local.kd
      .gfx1250_revision: B0
      .sgpr_count: 32
      .vgpr_count: 22
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
