// COM: One function owns a 24-byte tail NOP sled. Its first DS2 replacement
// COM: consumes 20 bytes, leaving exactly one dword. A second DS2 site needs a
// COM: far trampoline whose forward edge uses that remaining dword as a branch
// COM: island. Gateway allocation must continue from the shared sled cursor,
// COM: not rediscover stale NOPs and overwrite the replacement body.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: recognized materialized PC transfer
// LOG: hotswap: safe far return: used local NOP sled
// LOG: hotswap: safe far return: reusing original site-dead s[104:105]
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <owned_tail_kernel>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM: ds_store_b64 v22, v[10:11] offset:856
// DISASM-NEXT: ds_store_b64 v22, v[18:19] offset:1080
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch
// DISASM-LABEL: <callee>:

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl owned_tail_kernel
.p2align 8
.type owned_tail_kernel,@function
owned_tail_kernel:
  ds_store_2addr_b64 v22, v[10:11], v[18:19] offset0:107 offset1:135

// One path reaches the materialized return without defining s[104:105], so
// the first DS2 cannot claim that pair as site-dead scratch. The other path
// reaches the far source, where the pair is defined before every exit.
  s_cbranch_scc1 .Lmaterialized_exit
  s_cbranch_scc1 .Lfar_resume
.Lfar_source:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lfar_resume:
  s_wait_dscnt 0
  s_mov_b64 s[104:105], 0
  .rept 15000
    s_mov_b32 s64, s65
  .endr

.Lmaterialized_exit:
  s_cmp_lg_u64 s[0:1], 0
  s_cmp_lg_u64 s[2:3], 0
  s_cmp_lg_u64 s[104:105], 0
  s_cmp_lg_u64 s[30:31], 0
  s_and_saveexec_b32 s0, vcc_lo
  s_get_pc_i64 s[2:3]
  s_add_nc_u64 s[2:3], s[2:3], 44
  ds_store_b64 v1, v[4:5]
  s_swap_pc_i64 s[30:31], s[2:3]
  s_setpc_b64 s[30:31]

// The body and gateway are two roles sharing one allocation cursor. The
// no-fallthrough return makes this a valid body-capable tail sled, and the
// function range owns every byte.
.Lowned_tail_sled:
.rept 6
  s_nop 0
.endr
.size owned_tail_kernel, .-owned_tail_kernel

.local callee
.type callee,@function
callee:
  s_setpc_b64 s[30:31]
.size callee, .-callee

// Keep the appended trampoline pool beyond direct s_branch reach. The owned
// tail remains within source reach and can bridge the far source to the pool.
.rept 25000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel owned_tail_kernel
  .amdhsa_next_free_vgpr 24
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: owned_tail_kernel
      .symbol: owned_tail_kernel.kd
      .sgpr_count: 106
      .vgpr_count: 24
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
