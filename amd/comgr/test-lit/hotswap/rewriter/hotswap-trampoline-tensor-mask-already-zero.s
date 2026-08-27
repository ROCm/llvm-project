// COM: A tensor descriptor whose base has a provably zero low half needs no
// COM: rewrite. Both CFG paths seed s4 with zero and then use only self-writes
// COM: that cannot introduce a low bit. Foreign reads after the tensor are
// COM: harmless because this proof changes neither the definition nor the
// COM: tensor. A variant with one nonzero seed must reject the no-op proof and
// COM: use the save/clear/tensor/restore fallback.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=ZERO %s
// ZERO: descriptor workgroup_mask is already zero on every path
// ZERO: hotswap: applied 0 instruction patches
// ZERO: RESULT: SUCCESS
// RUN: cmp %t.elf %t.out.elf

// COM: An unresolved set-PC is a possible alternate entry to the tensor. The
// COM: modeled zero path cannot prove anything about s4 on that hidden edge.
// RUN: sed 's/^\.set opaque_tensor_entry, 0$/.set opaque_tensor_entry, 1/' \
// RUN:   %s > %t.opaque.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.opaque.s -o %t.opaque.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.opaque.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.opaque.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=NONZERO %s

// RUN: sed 's/^\.set nonzero_path, 0$/.set nonzero_path, 1/' \
// RUN:   %s > %t.nonzero.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.nonzero.s -o %t.nonzero.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nonzero.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.nonzero.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=NONZERO %s
// NONZERO-NOT: descriptor workgroup_mask is already zero on every path
// NONZERO: tensor_load_to_lds: s4 live, save/restore via s12
// NONZERO: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.nonzero.out.elf \
// RUN:   | %FileCheck --check-prefix=NONZERO-DIS %s
// NONZERO-DIS-LABEL: <test_tensor_mask_already_zero>:
// NONZERO-DIS: s_and_b32 s4, s4, 0xfff7ffff
// NONZERO-DIS: s_mov_b32 s12, s4
// NONZERO-DIS-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// NONZERO-DIS-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// NONZERO-DIS-NEXT: s_mov_b32 s4, s12

.set nonzero_path, 0
.set opaque_tensor_entry, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_already_zero
.p2align 8
.type test_tensor_mask_already_zero,@function
test_tensor_mask_already_zero:
.if opaque_tensor_entry
  s_cbranch_scc0 .Lopaque_entry
.endif
  s_cbranch_execz .Lsecond
  s_mov_b32 s4, 0
  s_branch .Lbuild
.Lsecond:
.if nonzero_path
  s_mov_b32 s4, 1
.else
  s_mov_b32 s4, 0
.endif
.Lbuild:
  s_and_b32 s4, s4, 0xfffcffff
  s_or_b32 s4, s4, 0x10000
  s_and_b32 s4, s4, 0xfff7ffff
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
  .rept 8
    s_nop 0
  .endr
.if opaque_tensor_entry
.Lopaque_entry:
  s_mov_b32 s4, 1
  s_set_pc_i64 s[10:11]
.endif
.Ltest_tensor_mask_already_zero_end:
.size test_tensor_mask_already_zero, .Ltest_tensor_mask_already_zero_end-test_tensor_mask_already_zero

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_already_zero
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_already_zero
      .symbol: test_tensor_mask_already_zero.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
