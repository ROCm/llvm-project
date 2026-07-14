// Exercise the tensor descriptor low16 must analysis independently of the
// canonical-delay and local-definition fallbacks. Positive cases prove the
// descriptor on every path. Negative cases omit the delay and include an
// intervening descriptor use so an unsound fact would incorrectly succeed.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.copy.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.copy.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.copy.out 2>&1 | %FileCheck --check-prefix=POS %s
// RUN: hotswap-rewrite %t.copy.out amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --check-idempotent | %FileCheck --check-prefix=IDEM %s
// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.merge.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.merge.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.merge.out 2>&1 | %FileCheck --check-prefix=POS %s
// RUN: hotswap-rewrite %t.merge.out amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --check-idempotent | %FileCheck --check-prefix=IDEM %s
// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.merge-clobber.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.merge-clobber.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=4 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.loop-clobber.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.loop-clobber.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=5 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.true16.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.true16.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=6 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.vopd.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.vopd.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.vopd.out 2>&1 | %FileCheck --check-prefix=POS %s
// RUN: hotswap-rewrite %t.vopd.out amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --check-idempotent | %FileCheck --check-prefix=IDEM %s
// RUN: %clang -x assembler-with-cpp -DCASE=7 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.mixed-vopd.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.mixed-vopd.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=8 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.exec-zero.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.exec-zero.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=9 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.intervening-use.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.intervening-use.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=10 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.shared.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.shared.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.shared.out 2>&1 | %FileCheck --check-prefix=SHARED %s
// RUN: hotswap-rewrite %t.shared.out amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --check-idempotent | %FileCheck --check-prefix=IDEM %s
// RUN: %clang -x assembler-with-cpp -DCASE=11 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.entry-loop.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.entry-loop.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=12 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.sgpr-overlap.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.sgpr-overlap.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --expect-status ERROR 2>&1 | %FileCheck --check-prefix=NEG %s
// RUN: %clang -x assembler-with-cpp -DCASE=13 -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.invariant-loop.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.invariant-loop.elf amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --output %t.invariant-loop.out 2>&1 | %FileCheck --check-prefix=POS %s
// RUN: hotswap-rewrite %t.invariant-loop.out amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 --strict-mode --check-idempotent | %FileCheck --check-prefix=IDEM %s

// POS: descriptor low16 already zero at 0x
// POS: RESULT: SUCCESS
// NEG-NOT: descriptor low16 already zero
// NEG: hotswap: error: tensor_load_to_lds at 0x
// NEG: RESULT: ERROR
// SHARED: masked local descriptor definition at 0x
// SHARED: reusing masked descriptor definition at 0x
// SHARED: RESULT: SUCCESS
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_must_cfg
.p2align 8
.type test_tensor_must_cfg,@function
test_tensor_must_cfg:
#if CASE == 1
  s_mov_b32 s20, 0
  s_mov_b32 s4, s20
#elif CASE == 2
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Lzero_path
  s_mov_b32 s4, 0
  s_branch .Ltensor
.Lzero_path:
  s_mov_b32 s4, 0
.Ltensor:
#elif CASE == 3
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Lunknown_path
  s_mov_b32 s4, 0
  s_branch .Ltensor
.Lunknown_path:
  s_mov_b32 s4, 1
.Ltensor:
#elif CASE == 4
  s_mov_b32 s4, 0
.Lloop:
  s_mov_b32 s4, s20
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Lloop
#elif CASE == 5
  v_mov_b32 v0, 0
  v_add_f16 v0.l, v1.l, v2.l
  v_readfirstlane_b32 s4, v0
  s_cmp_eq_u32 s4, 0
#elif CASE == 6
  v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0
  v_readfirstlane_b32 s4, v0
#elif CASE == 7
  v_dual_mov_b32 v0, 0 :: v_dual_add_nc_u32 v1, v2, v3
  v_readfirstlane_b32 s4, v0
  s_cmp_eq_u32 s4, 0
#elif CASE == 8
  s_mov_b64 exec, 0
  v_mov_b32 v0, 0
  v_readfirstlane_b32 s4, v0
  s_mov_b64 exec, -1
  s_cmp_eq_u32 s4, 0
#elif CASE == 9
  v_readfirstlane_b32 s4, v0
  s_mov_b32 s20, s4
#elif CASE == 10
  v_readfirstlane_b32 s4, v0
  tensor_load_to_lds s[24:27], s[4:11]
  s_nop 0
#elif CASE == 11
  v_mov_b32 v0, 0
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 .Lentry_loop_done
  s_mov_b64 exec, 0
  s_branch test_tensor_must_cfg
.Lentry_loop_done:
  v_readfirstlane_b32 s4, v0
  s_cmp_eq_u32 s4, 0
#elif CASE == 12
  v_mov_b32 v0, 0
  v_readfirstlane_b32 s4, v0
  s_mov_b64 s[4:5], s[20:21]
  s_cmp_eq_u32 s4, 0
#elif CASE == 13
  s_mov_b32 s4, 0
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 test_tensor_must_cfg
#else
#error unsupported CASE
#endif
  tensor_load_to_lds s[24:27], s[4:11]
  s_endpgm
.Ltest_tensor_must_cfg_end:
.size test_tensor_must_cfg, .Ltest_tensor_must_cfg_end-test_tensor_must_cfg

.rodata
.p2align 8
.amdhsa_kernel test_tensor_must_cfg
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 28
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_must_cfg
      .symbol: test_tensor_must_cfg.kd
      .sgpr_count: 28
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
