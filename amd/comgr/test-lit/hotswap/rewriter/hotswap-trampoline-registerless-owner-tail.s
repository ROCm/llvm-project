// COM: A far trampoline can have a directly reachable forward edge while its
// COM: longer copied body makes the return edge just exceed s_branch range.
// COM: With no scratch pair at the original source, the return allocator must
// COM: promote safe straight-line windows across a gap wider than two s_branch
// COM: spans. Each demand-sized promoted window supplies two suffix dwords:
// COM: the return chain consumes the first and the registerless forward chain
// COM: reuses the same source window's second slot. All numbered SGPRs are live
// COM: after every window, so both promotions must use proven-dead VCC.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: safe far return: no register pair at
// LOG: hotswap: promoted safe straight-line source at {{.*}} to provide return-capacity relay window {{.*}} with 2 slot(s) using VCC
// LOG: hotswap: promoted safe straight-line source at {{.*}} to provide return-capacity relay window {{.*}} with 2 slot(s) using VCC
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG: hotswap: assigned 1 return s_branch island chain(s)
// LOG: hotswap: promoted 2 safe straight-line source(s) for branch-island capacity
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d --no-symbolize-operands %t.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <registerless_source>:
// DISASM-NEXT: s_branch
// DISASM-LABEL: <promotion_candidate>:
// DISASM-NEXT: s_mov_b32 s0, s1
// DISASM-NEXT: s_get_pc_i64 vcc
// DISASM-NEXT: s_add_nc_u64 vcc, vcc,
// DISASM-NEXT: s_set_pc_i64 vcc
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_mov_b32 s0, s8
// DISASM-LABEL: <promotion_candidate_far>:
// DISASM-NEXT: s_mov_b32 s0, s17
// DISASM-NEXT: s_get_pc_i64 vcc
// DISASM-NEXT: s_add_nc_u64 vcc, vcc,
// DISASM-NEXT: s_set_pc_i64 vcc
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_mov_b32 s0, s24

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local targeter
.type targeter,@function
targeter:
  s_branch registerless_source_after
  s_endpgm
  .size targeter, .-targeter

// Put the source close to the positive s_branch boundary of the page-aligned
// trampoline pool. The count is tuned together with the tail padding below.
.rept 690
  s_mov_b32 s0, s1
.endr

.globl registerless_source
.type registerless_source,@function
registerless_source:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
registerless_source_after:
  // Keep every numbered pair live-in so neither local nor object-wide
  // scratch analysis can manufacture a set-PC return pair.
  .irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105
    s_mov_b32 s\reg, s\reg
  .endr
  s_mov_b32 s0, vcc_lo
  s_endpgm
  .size registerless_source, .-registerless_source

// Keep the candidate near the midpoint of the source-to-pool span.
.rept 15400
  s_mov_b32 s0, s1
.endr

.globl promotion_candidate
.type promotion_candidate,@function
promotion_candidate:
  // Distinct source operands make each six-instruction relocated body
  // identifiable in the appended pool.
  s_mov_b32 s0, s1
  s_mov_b32 s0, s2
  s_mov_b32 s0, s3
  s_mov_b32 s0, s4
  s_mov_b32 s0, s5
  s_mov_b32 s0, s6
  s_mov_b32 s0, s7
  s_mov_b32 s0, s8
  s_mov_b32 s0, s9
  s_mov_b32 s0, s10
  s_mov_b32 s0, s11
  s_mov_b32 s0, s12
  s_mov_b32 s0, s13
  s_mov_b32 s0, s14
  s_mov_b32 s0, s15
  s_mov_b32 s0, s16
  // The relocatable prefix does not need incoming VCC, while every numbered
  // pair remains live at its continuation. This selects dead VCC conservatively.
  .irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105
    s_mov_b32 s\reg, s\reg
  .endr
  s_endpgm
  .size promotion_candidate, .-promotion_candidate

// The second promotion region is over one branch span from the first, and the
// final pool is over one further span away.
.rept 25000
  s_mov_b32 s0, s1
.endr

.globl promotion_candidate_far
.type promotion_candidate_far,@function
promotion_candidate_far:
  s_mov_b32 s0, s17
  s_mov_b32 s0, s18
  s_mov_b32 s0, s19
  s_mov_b32 s0, s20
  s_mov_b32 s0, s21
  s_mov_b32 s0, s22
  s_mov_b32 s0, s23
  s_mov_b32 s0, s24
  s_mov_b32 s0, s25
  s_mov_b32 s0, s26
  s_mov_b32 s0, s27
  s_mov_b32 s0, s28
  s_mov_b32 s0, s29
  s_mov_b32 s0, s30
  s_mov_b32 s0, s31
  s_mov_b32 s0, s32
  .irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105
    s_mov_b32 s\reg, s\reg
  .endr
  s_endpgm
  .size promotion_candidate_far, .-promotion_candidate_far

.rept 25000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel registerless_source
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: registerless_source
      .symbol: registerless_source.kd
      .sgpr_count: 106
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
