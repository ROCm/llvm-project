// COM: Verify that anonymous zero-filled alignment after an explicitly-sized
// COM: kernel is usable as a function-owned code cave. Kernel A references
// COM: every numbered SGPR pair, so no set-PC scratch pair is available. Its
// COM: two DS2 loads therefore use 20-byte compact split-load bodies in the
// COM: anonymous .p2align gap before kernel B. Kernel B is not part of A's
// COM: cave and must remain unchanged.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Both 8-byte DS2 sites use a short redirect and retain the split drain in
// COM: their original second dword.
// DISASM-LABEL: <test_ds_kernel_tail_cave_a>:
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  s_wait_dscnt 0x0
// DISASM:       s_endpgm

// COM: The compact bodies omit the drain and return to the drain retained at
// COM: original+4. The first body preserves natural load order.
// DISASM:       ds_load_b32 v0, v4 offset:4
// DISASM-NEXT:  ds_load_b32 v1, v4 offset:8
// DISASM-NEXT:  s_branch

// COM: The second body's address v2 aliases its first destination, so it must
// COM: issue the v3 half first. Its return is also a direct short branch.
// DISASM-NEXT:  ds_load_b32 v3, v2 offset:16
// DISASM-NEXT:  ds_load_b32 v2, v2 offset:12
// DISASM-NEXT:  s_branch

// COM: Kernel B starts at the next alignment boundary and remains unchanged.
// DISASM-LABEL: <test_ds_kernel_tail_cave_b>:
// DISASM-NEXT:  s_mov_b32 s0, 0x1234
// DISASM-NEXT:  s_endpgm

// COM: Rewriting the result again must be byte-identical.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_ds_kernel_tail_cave_a
.p2align 8
.type test_ds_kernel_tail_cave_a,@function
test_ds_kernel_tail_cave_a:
  ds_load_2addr_b32 v[0:1], v4 offset0:1 offset1:2
  ds_load_2addr_b32 v[2:3], v2 offset0:3 offset1:4
  // Keep every numbered SGPR pair live across both required sites. This makes
  // the anonymous alignment cave, rather than a set-PC trampoline, mandatory.
  // Every high caller-clobbered pair is read before any defining instruction;
  // the comparisons also make every pair globally used.
  s_cmp_lg_u64 s[0:1], 0
  s_cmp_lg_u64 s[2:3], 0
  s_cmp_lg_u64 s[4:5], 0
  s_cmp_lg_u64 s[6:7], 0
  s_cmp_lg_u64 s[8:9], 0
  s_cmp_lg_u64 s[10:11], 0
  s_cmp_lg_u64 s[12:13], 0
  s_cmp_lg_u64 s[14:15], 0
  s_cmp_lg_u64 s[16:17], 0
  s_cmp_lg_u64 s[18:19], 0
  s_cmp_lg_u64 s[20:21], 0
  s_cmp_lg_u64 s[22:23], 0
  s_cmp_lg_u64 s[24:25], 0
  s_cmp_lg_u64 s[26:27], 0
  s_cmp_lg_u64 s[28:29], 0
  s_cmp_lg_u64 s[30:31], 0
  s_cmp_lg_u64 s[32:33], 0
  s_cmp_lg_u64 s[34:35], 0
  s_cmp_lg_u64 s[36:37], 0
  s_cmp_lg_u64 s[38:39], 0
  s_cmp_lg_u64 s[40:41], 0
  s_cmp_lg_u64 s[42:43], 0
  s_cmp_lg_u64 s[44:45], 0
  s_cmp_lg_u64 s[46:47], 0
  s_cmp_lg_u64 s[48:49], 0
  s_cmp_lg_u64 s[50:51], 0
  s_cmp_lg_u64 s[52:53], 0
  s_cmp_lg_u64 s[54:55], 0
  s_cmp_lg_u64 s[56:57], 0
  s_cmp_lg_u64 s[58:59], 0
  s_cmp_lg_u64 s[60:61], 0
  s_cmp_lg_u64 s[62:63], 0
  s_cmp_lg_u64 s[64:65], 0
  s_cmp_lg_u64 s[66:67], 0
  s_cmp_lg_u64 s[68:69], 0
  s_cmp_lg_u64 s[70:71], 0
  s_cmp_lg_u64 s[72:73], 0
  s_cmp_lg_u64 s[74:75], 0
  s_cmp_lg_u64 s[76:77], 0
  s_cmp_lg_u64 s[78:79], 0
  s_cmp_lg_u64 s[80:81], 0
  s_cmp_lg_u64 s[82:83], 0
  s_cmp_lg_u64 s[84:85], 0
  s_cmp_lg_u64 s[86:87], 0
  s_cmp_lg_u64 s[88:89], 0
  s_cmp_lg_u64 s[90:91], 0
  s_cmp_lg_u64 s[92:93], 0
  s_cmp_lg_u64 s[94:95], 0
  s_cmp_lg_u64 s[96:97], 0
  s_cmp_lg_u64 s[98:99], 0
  s_cmp_lg_u64 s[100:101], 0
  s_cmp_lg_u64 s[102:103], 0
  s_cmp_lg_u64 s[104:105], 0
  // Cross the next 256-byte boundary so the following alignment gap is large
  // enough for both 20-byte compact bodies.
  s_cmp_lg_u64 s[0:1], 0
  s_cmp_lg_u64 s[2:3], 0
  s_cmp_lg_u64 s[4:5], 0
  s_cmp_lg_u64 s[6:7], 0
  s_cmp_lg_u64 s[8:9], 0
  s_cmp_lg_u64 s[10:11], 0
  s_cmp_lg_u64 s[12:13], 0
  s_endpgm
.size test_ds_kernel_tail_cave_a, .-test_ds_kernel_tail_cave_a

.globl test_ds_kernel_tail_cave_b
.p2align 8, 0
.type test_ds_kernel_tail_cave_b,@function
test_ds_kernel_tail_cave_b:
  s_mov_b32 s0, 0x1234
  s_endpgm
  // Keep the appended pool outside s_branch reach so kernel A must use its
  // anonymous alignment cave once every set-PC scratch pair is rejected.
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.size test_ds_kernel_tail_cave_b, .-test_ds_kernel_tail_cave_b

.rodata
.p2align 8
.amdhsa_kernel test_ds_kernel_tail_cave_a
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdhsa_kernel test_ds_kernel_tail_cave_b
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds_kernel_tail_cave_a
      .symbol: test_ds_kernel_tail_cave_a.kd
      .sgpr_count: 106
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds_kernel_tail_cave_b
      .symbol: test_ds_kernel_tail_cave_b.kd
      .sgpr_count: 1
      .vgpr_count: 0
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
