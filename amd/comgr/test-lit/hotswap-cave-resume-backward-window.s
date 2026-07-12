// COM: A branch-back from an immediate NOP-sled rewrite must remain a valid
// COM: control-flow target while later far sites are finalized. In particular,
// COM: a backward set-PC source window must not begin at that return address.
// COM: The first required rewrite consumes 24 bytes of this function's 32-byte
// COM: sled. The later far DS2 rewrite cannot grow forward across s_endpgm and
// COM: must use the following function-alignment cave as a relay instead of
// COM: growing backward over the first rewrite's return target.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=API,LOG %s
// LOG: set-PC forward site 0x34 reserved relay
// LOG-NOT: expanded backward to 0x2C
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The local WMMA body returns to the first instruction following the
// COM: original WMMA. Those two SALU instructions must remain intact. The DS2
// COM: site uses get-PC plus a short branch into the remaining relay slot.
// DISASM-LABEL: <test_cave_resume_backward_window>:
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT:  s_branch
// DISASM:       s_branch
// DISASM-NEXT:  s_nop 0
// DISASM-NEXT:  s_mov_b32 s20, s21
// DISASM-NEXT:  s_mov_b32 s22, s23
// DISASM-NEXT:  s_get_pc_i64
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  s_endpgm

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cave_resume_backward_window
.p2align 8
.type test_cave_resume_backward_window,@function
test_cave_resume_backward_window:
  // Exactly 32 bytes: the WMMA cave uses 24. The guard inserted at the first
  // word preserves normal entry.
  .rept 8
    s_nop 0
  .endr
  s_mov_b32 s18, s19
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_mov_b32 s20, s21
  s_mov_b32 s22, s23
  ds_load_2addr_b32 v[0:1], v4 offset0:1 offset1:2
  s_endpgm
.Ltest_cave_resume_backward_window_end:
.size test_cave_resume_backward_window, .Ltest_cave_resume_backward_window_end-test_cave_resume_backward_window

// The 256-byte function-alignment gap is deliberately outside the function,
// so the original forward window still stops at s_endpgm. Finalization should
// reserve only the small relay it needs from this proven tail cave.
.p2align 8, 0
.globl test_cave_resume_filler
.type test_cave_resume_filler,@function
test_cave_resume_filler:
  // Keep the appended pool outside direct s_branch reach without introducing
  // a relocatable forward instruction at the DS2 site.
  .rept 40000
    s_mov_b32 s24, s25
  .endr
  s_setpc_b64 s[30:31]
.Ltest_cave_resume_filler_end:
.size test_cave_resume_filler, .Ltest_cave_resume_filler_end-test_cave_resume_filler

.rodata
.p2align 8
.amdhsa_kernel test_cave_resume_backward_window
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 26
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cave_resume_backward_window
      .symbol: test_cave_resume_backward_window.kd
      .sgpr_count: 26
      .vgpr_count: 40
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
