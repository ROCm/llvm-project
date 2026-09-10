// COM: A far eight-byte trampoline site may tentatively reserve one SGPR to
// COM: preserve live wave32 VCC_LO. Direct control-flow and a following branch
// COM: prevent expansion to the required 12-byte source landing, so the plan
// COM: must downgrade cleanly to the registerless island route. Its second
// COM: source dword is the owner-reserved final return relay, so it branches
// COM: to the continuation instead of remaining padding.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: safe far return: deferring live wave32 VCC_LO preservation in s105
// LOG: hotswap: deferred live-VCC preservation at 0x{{[0-9A-F]+}} fell back to a registerless far return
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_deferred_vcc>:
// DISASM-NEXT: s_branch 0
// DISASM-NEXT: s_mov_b32 s104, 0
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_cbranch_vccz

// RUN: %llvm-readelf --notes %t.out.elf \
// RUN:   | %FileCheck --check-prefix=METADATA %s
// METADATA: .name:           test_deferred_vcc
// METADATA: .sgpr_count:     105

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_deferred_vcc
.p2align 8
.type test_deferred_vcc,@function
test_deferred_vcc:
  s_branch 0
  s_mov_b32 s104, 0
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_cbranch_vccz 0
.irp live_reg, s0, s2, s4, s6, s8, s10, s12, s14, s16, s18, s20, s22, s24, s26, s28, s30, s32, s34, s36, s38, s40, s42, s44, s46, s48, s50, s52, s54, s56, s58, s60, s62, s64, s66, s68, s70, s72, s74, s76, s78, s80, s82, s84, s86, s88, s90, s92, s94, s96, s98, s100, s102, s104
  s_mov_b32 s1, \live_reg
.endr
  s_endpgm
.size test_deferred_vcc, .-test_deferred_vcc

.type gateway_0,@function
gateway_0:
  s_endpgm
.size gateway_0, .-gateway_0
.fill 32, 1, 0

.rept 20000
  s_mov_b32 s0, s1
.endr

.type gateway_1,@function
gateway_1:
  s_endpgm
.size gateway_1, .-gateway_1
.fill 32, 1, 0

.rept 20000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_deferred_vcc
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 105
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_deferred_vcc
      .symbol: test_deferred_vcc.kd
      .sgpr_count: 105
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
