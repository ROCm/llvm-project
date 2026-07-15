// A WMMA co-execution requirement and an ordinary FP8 CLAMP correction can
// target the same instruction. The central replacement owner must prepend the
// required v_nops to the FP8 emulation body instead of queuing two competing
// source rewrites. Exercise both appended-trampoline and owned-sled placement.

// RUN: %clang -x assembler-with-cpp -DCASE=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.trampoline.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.trampoline.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.trampoline.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,PACK %s
// RUN: %llvm-objdump -d %t.trampoline.out.elf \
// RUN:   | %FileCheck --check-prefix=TRAMPOLINE %s
// RUN: hotswap-rewrite %t.trampoline.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// RUN: %clang -x assembler-with-cpp -DCASE=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.sled.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.sled.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.sled.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,PACK %s
// RUN: %llvm-objdump -d %t.sled.out.elf \
// RUN:   | %FileCheck --check-prefix=SLED %s
// RUN: hotswap-rewrite %t.sled.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// The unpack patch calls emitToTrampoline directly rather than the sled-aware
// emitReplacementCode entry point. It must use the same single composition
// transaction and must not prepend the requirement twice.
// RUN: %clang -x assembler-with-cpp -DCASE=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.direct.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.direct.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.direct.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,UNPACK %s
// RUN: %llvm-objdump -d %t.direct.out.elf \
// RUN:   | %FileCheck --check-prefix=DIRECT %s
// RUN: hotswap-rewrite %t.direct.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s

// API: WMMA co-exec hazard at 0x{{[0-9A-F]+}}: v_wmma_i32_16x16x64_iu8 needs 8 v_nops
// API: WMMA co-exec validation: 1 hazards (1 WMMA instructions scanned)
// API: WMMA co-exec requirement composed into replacement at 0x{{[0-9A-F]+}} (8 leading v_nop(s))
// PACK: cvt_pk_fp8_f32: patched CLAMP=1 (E5M3) at offset 0x{{[0-9A-F]+}}
// UNPACK: cvt_f32_fp8: patched CLAMP=1 (E5M3) at offset 0x{{[0-9A-F]+}}
// API: applied 2 instruction patches
// API: RESULT: SUCCESS
// IDEM: IDEMPOTENT: YES

// TRAMPOLINE-LABEL: <test_wmma_hazard_composition>:
// TRAMPOLINE-NEXT: v_wmma_i32_16x16x64_iu8
// TRAMPOLINE-NEXT: s_branch
// TRAMPOLINE-NEXT: s_nop 0
// TRAMPOLINE-NEXT: s_endpgm
// TRAMPOLINE-COUNT-8: v_nop
// TRAMPOLINE-NEXT: s_mov_b32
// TRAMPOLINE-NEXT: v_and_b32{{.*}}0x7fffffff, v1

// SLED-LABEL: <test_wmma_hazard_composition>:
// SLED-NEXT: v_wmma_i32_16x16x64_iu8
// SLED-NEXT: s_branch
// SLED-NEXT: s_nop 0
// SLED-NEXT: s_endpgm
// SLED-COUNT-8: v_nop
// SLED-NEXT: s_mov_b32
// SLED-NEXT: v_and_b32{{.*}}0x7fffffff, v1

// DIRECT-LABEL: <test_wmma_hazard_composition>:
// DIRECT-NEXT: v_wmma_i32_16x16x64_iu8
// DIRECT-NEXT: s_branch
// DIRECT-NEXT: s_nop 0
// DIRECT-NEXT: s_endpgm
// DIRECT-COUNT-8: v_nop
// DIRECT-NEXT: s_mov_b32
// DIRECT-NEXT: v_and_b32{{.*}}0xff, v1

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wmma_hazard_composition
.p2align 8
.type test_wmma_hazard_composition,@function
test_wmma_hazard_composition:
  v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
#if CASE == 3
  v_cvt_f32_fp8 v16, v1 clamp
#else
  v_cvt_pk_fp8_f32 v16, v1, v2 clamp
#endif
  s_endpgm
#if CASE == 2
  .rept 96
  s_nop 0
  .endr
#elif CASE != 1 && CASE != 3
  .error "CASE must select a test body"
#endif
.Ltest_wmma_hazard_composition_end:
.size test_wmma_hazard_composition, .Ltest_wmma_hazard_composition_end-test_wmma_hazard_composition

.rodata
.p2align 8
.amdhsa_kernel test_wmma_hazard_composition
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_hazard_composition
      .symbol: test_wmma_hazard_composition.kd
      .sgpr_count: 2
      .vgpr_count: 24
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
