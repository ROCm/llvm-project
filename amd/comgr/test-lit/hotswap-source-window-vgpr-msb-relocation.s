// COM: Straight-line source growth may relocate a complete s_set_vgpr_msb
// COM: instruction when doing so preserves its dynamic order. A known entry
// COM: may become the new window start, but can never become its interior.

// RUN: %clang -x assembler-with-cpp -DFORWARD \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   -nostdlib %s -o %t.forward.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.forward.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.forward.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=FORWARD-LOG %s
// FORWARD-LOG-NOT: hotswap: error:
// FORWARD-LOG: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.forward.out.elf \
// RUN:   | %FileCheck --check-prefix=FORWARD %s
// FORWARD-LABEL: <mode_setter_source_window>:
// FORWARD-NEXT: s_set_vgpr_msb 1
// FORWARD-NEXT: s_get_pc_i64
// FORWARD-NEXT: s_add_nc_u64
// FORWARD-NEXT: s_set_pc_i64
// FORWARD: ds_load_b64 v[2:3], v52 offset:1224
// FORWARD-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// FORWARD-NEXT: s_wait_dscnt 0x0
// FORWARD-NEXT: s_set_vgpr_msb 0x100
// FORWARD-NEXT: ds_load_b64 v[28:29], v87 offset:17328

// RUN: %clang -x assembler-with-cpp -DBACKWARD \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   -nostdlib %s -o %t.backward.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.backward.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.backward.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=BACKWARD-LOG %s
// BACKWARD-LOG-NOT: hotswap: error:
// BACKWARD-LOG: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.backward.out.elf \
// RUN:   | %FileCheck --check-prefix=BACKWARD %s
// BACKWARD-LABEL: <mode_setter_source_window>:
// BACKWARD-NEXT: s_set_vgpr_msb 1
// BACKWARD-NEXT: s_get_pc_i64
// BACKWARD-NEXT: s_add_nc_u64
// BACKWARD-NEXT: s_set_pc_i64
// BACKWARD: ds_load_b64 v[28:29], v87 offset:17328
// BACKWARD-NEXT: s_set_vgpr_msb 0x100
// BACKWARD-NEXT: ds_load_b64 v[2:3], v52 offset:1224
// BACKWARD-NEXT: ds_load_b64 v[4:5], v52 offset:1288
// BACKWARD-NEXT: s_wait_dscnt 0x0

// RUN: %clang -x assembler-with-cpp -DTARGETED \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   -nostdlib %s -o %t.targeted.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.targeted.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.targeted.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=TARGETED-LOG %s
// TARGETED-LOG-NOT: hotswap: error:
// TARGETED-LOG: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.targeted.out.elf \
// RUN:   | %FileCheck --check-prefix=TARGETED %s
// TARGETED: s_set_vgpr_msb 0x100
// TARGETED: s_call_i64 s[10:11],

// RUN: hotswap-rewrite %t.forward.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl mode_setter_source_window
.p2align 8
.type mode_setter_source_window,@function
mode_setter_source_window:
#if defined(FORWARD)
  s_set_vgpr_msb 1
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
  s_set_vgpr_msb 0x100
  ds_load_b64 v[28:29], v87 offset:17328
#elif defined(BACKWARD)
  s_set_vgpr_msb 1
  ds_load_b64 v[28:29], v87 offset:17328
  s_set_vgpr_msb 0x100
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
#elif defined(TARGETED)
  s_branch mode_setter_site
  s_mov_b32 s4, s5
  s_set_vgpr_msb 0x100
mode_setter_site:
  ds_load_2addr_b64 v[2:5], v52 offset0:153 offset1:161
#endif
  s_endpgm
.Lmode_setter_source_window_end:
.size mode_setter_source_window, .Lmode_setter_source_window_end-mode_setter_source_window

#ifdef TARGETED
// The targeted patch site blocks backward growth because it would become an
// interior entry. This certified external padding supplies a gateway instead.
.rept 8
  s_nop 0
.endr
#endif

// Keep the appended trampoline pool beyond direct s_branch reach.
.rept 40000
  s_mov_b32 s8, s9
.endr

.rodata
.p2align 8
.amdhsa_kernel mode_setter_source_window
  .amdhsa_next_free_vgpr 88
  .amdhsa_next_free_sgpr 10
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: mode_setter_source_window
      .symbol: mode_setter_source_window.kd
      .sgpr_count: 10
      .vgpr_count: 88
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
