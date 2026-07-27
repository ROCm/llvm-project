// COM: Compact DS2 continuations are a scarce fallback for far sites without
// COM: any safe SGPR pair or dead VCC. Scratch-backed sites should keep using
// COM: the ordinary trampoline path so they do not consume set-PC-sized local
// COM: padding needed by object-wide gateway planning.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: safe far return: no register pair at 0x
// LOG: hotswap: ds_2addr: used compact continuation at 0x{{.*}}with displaced v_ashrrev_i32
// LOG: hotswap: safe far return: no register pair at 0x
// LOG: hotswap: ds_2addr: used compact continuation at 0x{{.*}}with displaced flat_load_b64
// LOG-NOT: hotswap: ds_2addr: used compact continuation at 0x
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_ds2_safe_far>:
// DISASM-NEXT: s_branch
// DISASM-LABEL: <test_ds2_registerless_far>:
// DISASM: ds_load_b32 v0, v2 offset:4
// DISASM-NEXT: s_branch
// DISASM-NEXT: ds_load_b32 v6, v8 offset:4
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM: ds_load_b32 v1, v2 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: v_ashrrev_i32_e32 v4, 31, v5
// DISASM-NEXT: s_branch
// DISASM: ds_load_b32 v7, v8 offset:12
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: flat_load_b64 v[10:11], v12, s[8:9] scale_offset
// DISASM-NEXT: s_branch
// DISASM-NOT: ds_load_2addr

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl test_ds2_safe_far
.p2align 8
.type test_ds2_safe_far,@function
test_ds2_safe_far:
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.size test_ds2_safe_far, .-test_ds2_safe_far

.globl test_ds2_registerless_far
.type test_ds2_registerless_far,@function
test_ds2_registerless_far:
  s_mov_b64 vcc, -1
  ds_load_2addr_b32 v[0:1], v2 offset0:1 offset1:3
  v_ashrrev_i32_e32 v4, 31, v5
  ds_load_2addr_b32 v[6:7], v8 offset0:1 offset1:3
  flat_load_b64 v[10:11], v12, s[8:9] scale_offset
  s_wait_dscnt 0x0
.irp live_reg, s0, s2, s4, s6, s8, s10, s12, s14, s16, s18, s20, s22, s24, s26, s28, s30, s32, s34, s36, s38, s40, s42, s44, s46, s48, s50, s52, s54, s56, s58, s60, s62, s64, s66, s68, s70, s72, s74, s76, s78, s80, s82, s84, s86, s88, s90, s92, s94, s96, s98, s100, s102, s104
  s_mov_b32 s1, \live_reg
.endr
  s_cbranch_vccz .Lregisterless_done
.Lregisterless_done:
  s_endpgm
.size test_ds2_registerless_far, .-test_ds2_registerless_far

// Three exact 20-byte external padding tails occupy the same reach bucket.
// The first remains the global-routing backbone; the safe DS2 must leave the
// surplus tails untouched so the registerless DS2 can claim one.
.type tail_guard_0,@function
tail_guard_0:
  s_endpgm
.size tail_guard_0, .-tail_guard_0
.fill 20, 1, 0
s_mov_b32 s0, s0

.type tail_guard_1,@function
tail_guard_1:
  s_endpgm
.size tail_guard_1, .-tail_guard_1
.fill 20, 1, 0
s_mov_b32 s0, s0

.type tail_guard_2,@function
tail_guard_2:
  s_endpgm
.size tail_guard_2, .-tail_guard_2
.fill 20, 1, 0
s_mov_b32 s0, s0

// This run contributes one 44-byte far-body partition and one 20-byte tail.
// The displaced 12-byte flat load makes a 28-byte continuation, proving that
// the larger compact form uses the far-body partition without a trampoline.
.type tail_guard_3,@function
tail_guard_3:
  s_endpgm
.size tail_guard_3, .-tail_guard_3
.fill 64, 1, 0
s_mov_b32 s0, s0

// Keep the appended trampoline pool outside signed s_branch reach.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_ds2_safe_far
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdhsa_kernel test_ds2_registerless_far
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 105
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2_safe_far
      .symbol: test_ds2_safe_far.kd
      .sgpr_count: 4
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2_registerless_far
      .symbol: test_ds2_registerless_far.kd
      .sgpr_count: 105
      .vgpr_count: 13
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
