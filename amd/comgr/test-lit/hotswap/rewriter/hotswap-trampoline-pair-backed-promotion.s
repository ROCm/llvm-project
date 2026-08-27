// COM: An eight-byte far patch source has room for only its one-dword forward
// COM: branch and one unreachable relay dword. A locally dead SGPR pair makes
// COM: the trampoline return pair-backed, but there is no variable-width
// COM: gateway and the source-to-pool corridor is wider than two s_branch
// COM: spans. The forward branch allocator must promote safe straight-line
// COM: source windows even though the owner has an SGPR-backed set-PC return.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: affine planner considering 1 pair-only source site(s)
// LOG: hotswap: ordinary gateway planner collected 1 pending site(s)
// LOG: hotswap: promoted safe straight-line source at {{.*}} to provide forward-capacity relay window
// LOG: hotswap: promoted safe straight-line source at {{.*}} to provide forward-capacity relay window
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG: hotswap: promoted 2 safe straight-line source(s) for branch-island capacity
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d --no-symbolize-operands %t.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <pair_backed_source>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch
// DISASM-LABEL: <promotion_candidate>:
// DISASM-NEXT: s_mov_b32 s0, s1
// DISASM-NEXT: s_mov_b32 s0, s2
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64
// DISASM: ds_load_b32
// DISASM-NEXT: ds_load_b32
// DISASM-NEXT: s_wait_dscnt
// DISASM-NEXT: s_get_pc_i64
// DISASM-NEXT: s_add_nc_u64
// DISASM-NEXT: s_set_pc_i64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local targeter
.type targeter,@function
targeter:
  s_branch pair_backed_source_after
  s_endpgm
.size targeter, .-targeter

// Put the source near the positive s_branch boundary of the page-aligned
// trampoline pool.
.rept 690
  s_mov_b32 s0, s1
.endr

.globl pair_backed_source
.type pair_backed_source,@function
pair_backed_source:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
pair_backed_source_after:
  // The numbered SGPRs are dead here, so the trampoline return uses a local
  // SGPR pair rather than the registerless return planner.
  s_mov_b32 s0, s1
  s_endpgm
.size pair_backed_source, .-pair_backed_source

// Only typed function bodies are promotion candidates. The untyped filler
// cannot become an accidental gateway.
.rept 15400
  s_mov_b32 s0, s1
.endr

.globl promotion_candidate
.type promotion_candidate,@function
promotion_candidate:
  s_mov_b32 s0, s1
  s_mov_b32 s0, s2
  s_mov_b32 s0, s3
  s_mov_b32 s0, s4
  s_mov_b32 s0, s5
  s_mov_b32 s0, s6
  s_mov_b32 s0, s7
  s_mov_b32 s0, s8
  s_endpgm
.size promotion_candidate, .-promotion_candidate

.rept 25000
  s_mov_b32 s0, s1
.endr

.globl promotion_candidate_far
.type promotion_candidate_far,@function
promotion_candidate_far:
  s_mov_b32 s0, s9
  s_mov_b32 s0, s10
  s_mov_b32 s0, s11
  s_mov_b32 s0, s12
  s_mov_b32 s0, s13
  s_mov_b32 s0, s14
  s_mov_b32 s0, s15
  s_mov_b32 s0, s16
  s_endpgm
.size promotion_candidate_far, .-promotion_candidate_far

.rept 25000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel pair_backed_source
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 17
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: pair_backed_source
      .symbol: pair_backed_source.kd
      .sgpr_count: 17
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
