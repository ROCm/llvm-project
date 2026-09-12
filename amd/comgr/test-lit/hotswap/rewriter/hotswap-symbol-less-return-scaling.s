// COM: Exercise symbol-less return-region inference with enough proven
// COM: regions, call sites and direct branch targets to expose the repeated
// COM: whole-object scans that made the proof quadratic in object size.
// COM: Each region is reachable only through a get-PC/add/swap-PC call that
// COM: supplies the exact link pair, carries no ELF symbol, and returns with
// COM: s_set_pc_i64 on that pair, so it is proven by
// COM: collectSymbolLessReturnRegions rather than by a function symbol.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf --entry-trampolines --strict-mode \
// RUN:   | %FileCheck %s
// CHECK: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
// The call sites and the regions they enter are emitted by two separate .rept
// blocks, so they are paired by an explicit counter rather than \@, which
// counts every macro expansion in the file.
.altmacro

// Every call materializes its callee address and supplies s[12:13] as the
// link pair. The callee carries no symbol, so the region can only be proven
// from the call site itself.
.macro EMIT_CALL_SITE IDX
.Lcall\IDX:
  s_get_pc_i64 s[10:11]
  s_add_nc_u64 s[10:11], s[10:11], .Lregion\IDX-(.Lcall\IDX+4)
  s_swap_pc_i64 s[12:13], s[10:11]
.endm

// The s_endpgm ahead of each entry proves the region cannot be entered by
// layout fallthrough from its predecessor. The region-internal branch adds a
// direct target that must be attributed to the owning region.
.macro EMIT_SYMBOL_LESS_REGION IDX
.Lregion\IDX:
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Ljoin\IDX
.Ljoin\IDX:
  s_nop 0
  s_nop 0
  s_set_pc_i64 s[12:13]
  s_endpgm
.endm

.globl symbol_less_return_scaling
.p2align 8
.type symbol_less_return_scaling,@function
symbol_less_return_scaling:
.set call_index, 0
.rept 1200
  EMIT_CALL_SITE %call_index
  .set call_index, call_index+1
.endr
  s_endpgm
.size symbol_less_return_scaling, .-symbol_less_return_scaling

.set region_index, 0
.rept 1200
  EMIT_SYMBOL_LESS_REGION %region_index
  .set region_index, region_index+1
.endr

.rodata
.p2align 8
.amdhsa_kernel symbol_less_return_scaling
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: symbol_less_return_scaling
      .symbol: symbol_less_return_scaling.kd
      .sgpr_count: 16
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
