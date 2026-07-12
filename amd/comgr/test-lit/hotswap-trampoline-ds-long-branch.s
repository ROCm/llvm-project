// COM: HSV-009 regression: gfx1250 A0 must not execute s_add_pc_i64. Two
// COM: adjacent required DS2 rewrites are placed more than 128 KB from the
// COM: appended pool. Their two 8-byte sites merge into the 16-byte forward
// COM: set-PC window. The pool returns through the same SCC-neutral sequence.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: set-PC forward windows: expanded 1 far site(s), merged 1 adjacent trampoline site(s), synthesized zero s_add_pc_i64
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOADD %s

// COM: The adjacent DS2 sites become one forward window. A 32-bit forward
// COM: literal makes this sequence exactly 16 bytes. The following SALU, drain,
// COM: and terminator remain at their linked PCs and are the return path.
// DISASM-LABEL: <test_ds2addr_far_adjacent>:
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM-NEXT: s_mov_b32 s20, s21
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm

// COM: The merged pool body contains both required expansions and one set-PC
// COM: return. The backward delta needs the full 20-byte form.
// DISASM:      ds_load_b64 v[0:1], v4 offset:512
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:1024
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_store_b32 v2, v0 offset:256
// DISASM-NEXT: ds_store_b32 v2, v1 offset:768
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64

// COM: Whole-object invariant, covering both source and pool disassembly.
// NOADD: file format elf64-amdgpu
// NOADD-NOT: s_add_pc_i64

// COM: The first rewrite consumes every B0-only DS2 opcode and the second is
// COM: byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2addr_far_adjacent
.p2align 8
.type test_ds2addr_far_adjacent,@function
test_ds2addr_far_adjacent:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
  s_mov_b32 s20, s21
  s_wait_dscnt 0x0
  s_endpgm
  // Non-NOP filler keeps the appended pool outside s_branch reach without
  // creating a local code cave.
  .rept 40000
    s_mov_b32 s20, s21
  .endr
.Ltest_ds2addr_far_adjacent_end:
.size test_ds2addr_far_adjacent, .Ltest_ds2addr_far_adjacent_end-test_ds2addr_far_adjacent

.rodata
.p2align 8
.amdhsa_kernel test_ds2addr_far_adjacent
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 22
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2addr_far_adjacent
      .symbol: test_ds2addr_far_adjacent.kd
      .sgpr_count: 22
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
