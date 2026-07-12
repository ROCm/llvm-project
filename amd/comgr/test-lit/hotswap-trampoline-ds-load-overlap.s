// COM: Verify that splitting DS 2-address loads preserves an address VGPR
// COM: that aliases a destination. Non-overlapping and second-half aliases
// COM: retain natural order. A first-half alias emits the second half first so
// COM: the first load result cannot replace the address before its second use.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_ds_load_overlap>:
// DISASM-NOT: ds_load_2addr

// COM: Non-overlapping b32 address: natural order.
// DISASM:      ds_load_b32 v0, v4 offset:4
// DISASM-NEXT: ds_load_b32 v1, v4 offset:8

// COM: First-half b32 alias: second half first.
// DISASM:      ds_load_b32 v3, v2 offset:16
// DISASM-NEXT: ds_load_b32 v2, v2 offset:12

// COM: Second-half b32 alias: natural order already preserves the address.
// DISASM:      ds_load_b32 v4, v5 offset:20
// DISASM-NEXT: ds_load_b32 v5, v5 offset:24

// COM: First-half b64 alias: second half first.
// DISASM:      ds_load_b64 v[8:9], v7 offset:16
// DISASM-NEXT: ds_load_b64 v[6:7], v7 offset:8

// COM: Second-half b64 alias: natural order.
// DISASM:      ds_load_b64 v[10:11], v12 offset:24
// DISASM-NEXT: ds_load_b64 v[12:13], v12 offset:32

// COM: The same first-half rule applies to stride64 b64 loads.
// DISASM:      ds_load_b64 v[16:17], v15 offset:1024
// DISASM-NEXT: ds_load_b64 v[14:15], v15 offset:512

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_ds_load_overlap
.p2align 8
.type test_ds_load_overlap,@function
test_ds_load_overlap:
  ds_load_2addr_b32 v[0:1], v4 offset0:1 offset1:2
  ds_load_2addr_b32 v[2:3], v2 offset0:3 offset1:4
  ds_load_2addr_b32 v[4:5], v5 offset0:5 offset1:6
  ds_load_2addr_b64 v[6:9], v7 offset0:1 offset1:2
  ds_load_2addr_b64 v[10:13], v12 offset0:3 offset1:4
  ds_load_2addr_stride64_b64 v[14:17], v15 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds_load_overlap_end:
.size test_ds_load_overlap, .Ltest_ds_load_overlap_end-test_ds_load_overlap

.rodata
.p2align 6
.amdhsa_kernel test_ds_load_overlap
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 0
  .amdhsa_user_sgpr_count 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 0
  .amdhsa_next_free_vgpr 18
  .amdhsa_next_free_sgpr 0
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
