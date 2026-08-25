// COM: A broad tensor function may contain a compiler-emitted reusable-PC
// COM: tail jump. Resolve only the exact adjacent get-PC/carry-add/set-PC
// COM: sequence to its in-range instruction boundary. Both the tensor path and
// COM: the jump target kill s4 after the tensor mask definition, so the
// COM: definition-time low16 clear is safe.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: cleared workgroup_mask at descriptor definition 0x{{[0-9A-F]+}} (s4)
// API-NOT: tensor_load_to_lds: s4
// API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DIS %s
// DIS-LABEL: <test_tensor_mask_local_setpc>:
// DIS: s_and_b32 s4, s4, 0xfff70000
// DIS-NOT: s_pack_hh_b32_b16
// DIS: tensor_load_to_lds s[0:3], s[4:11]
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.set wrong_pair, 0
.set intervening_def, 0
.set outside_target, 0
.set interior_entry, 0
.set declared_entry, 0

// COM: A wrong transfer pair, an intervening target-pair definition, or an
// COM: otherwise exact sequence whose target is outside the function must
// COM: remain opaque. Each variant therefore rejects the definition rewrite
// COM: and uses the at-site fallback.
// RUN: sed 's/^\.set wrong_pair, 0$/.set wrong_pair, 1/' \
// RUN:   %s > %t.wrong-pair.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.wrong-pair.s -o %t.wrong-pair.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.wrong-pair.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.wrong-pair.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OPAQUE %s
// RUN: sed 's/^\.set intervening_def, 0$/.set intervening_def, 1/' \
// RUN:   %s > %t.intervening.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.intervening.s -o %t.intervening.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.intervening.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.intervening.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OPAQUE %s
// RUN: sed 's/^\.set outside_target, 0$/.set outside_target, 1/' \
// RUN:   %s > %t.outside.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.outside.s -o %t.outside.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.outside.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.outside.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OPAQUE %s
// RUN: sed 's/^\.set interior_entry, 0$/.set interior_entry, 1/' \
// RUN:   %s > %t.interior.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.interior.s -o %t.interior.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.interior.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.interior.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OPAQUE %s
// RUN: sed 's/^\.set declared_entry, 0$/.set declared_entry, 1/' \
// RUN:   %s > %t.declared.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib \
// RUN:   %t.declared.s -o %t.declared.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.declared.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.declared.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=OPAQUE %s
// OPAQUE-NOT: cleared workgroup_mask at descriptor definition
// OPAQUE: tensor_load_to_lds: s4 dead, no save/restore needed
// OPAQUE: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_mask_local_setpc
.p2align 8
.type test_tensor_mask_local_setpc,@function
test_tensor_mask_local_setpc:
  s_and_b32 s4, s4, 0xfff7ffff
  s_cmp_eq_u32 s0, s0
.if interior_entry
  s_cbranch_scc0 .Ladd_low
.endif
  s_cbranch_execz .Ltensor
.Lgetpc:
  s_get_pc_i64 s[70:71]
.if outside_target
  s_add_co_i32 s72, .Loutside_target-(.Lgetpc+4)-4, 4
.else
  s_add_co_i32 s72, .Ltarget-(.Lgetpc+4)-4, 4
.endif
.Ladd_low:
.if declared_entry
.globl test_tensor_mask_local_setpc_interior
.type test_tensor_mask_local_setpc_interior,@function
test_tensor_mask_local_setpc_interior:
.endif
  s_add_co_u32 s70, s70, s72
.if declared_entry
.size test_tensor_mask_local_setpc_interior, .-test_tensor_mask_local_setpc_interior
.endif
  s_add_co_ci_u32 s71, s71, 0
.if intervening_def
  s_mov_b32 s70, 0
.endif
.if wrong_pair
  s_set_pc_i64 s[74:75]
.else
  s_set_pc_i64 s[70:71]
.endif
.Ltensor:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s4, 0
  s_cmp_eq_u32 s0, s0
  s_endpgm
.Ltarget:
  s_mov_b32 s4, 0
  s_cmp_eq_u32 s0, s0
  s_endpgm
.Ltest_tensor_mask_local_setpc_end:
.size test_tensor_mask_local_setpc, .Ltest_tensor_mask_local_setpc_end-test_tensor_mask_local_setpc

.Loutside:
  s_nop 0
.Loutside_target:
  s_endpgm

.rodata
.p2align 8
.amdhsa_kernel test_tensor_mask_local_setpc
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 76
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_mask_local_setpc
      .symbol: test_tensor_mask_local_setpc.kd
      .sgpr_count: 76
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
