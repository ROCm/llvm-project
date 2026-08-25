// A closed, compiler-emitted carry-chain set-PC later in the same function is
// understood by DirectControlFlow but not by whole-function VGPR-MSB mode
// recovery. The cross-bank DS2 still has a locally provable mode because its
// exact setter is adjacent and no control-flow or declared entry bypasses it.

// RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.local.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.local.o -o %t.local.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.local.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.local.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOCAL-LOG,LOCAL-API %s
// LOCAL-LOG-NOT: unresolved call target
// LOCAL-LOG-NOT: unresolved control-flow target
// LOCAL-API-NOT: hotswap: error:
// LOCAL-API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.local.out.elf \
// RUN:   | %FileCheck --check-prefix=LOCAL-DISASM %s
// LOCAL-DISASM-LABEL: <test_ds_vgpr_msb_local_closed>:
// LOCAL-DISASM:      s_set_vgpr_msb 0x480
// LOCAL-DISASM-NEXT: s_branch
// LOCAL-DISASM:      ds_load_b64 v[254:255]
// LOCAL-DISASM-NEXT: s_set_vgpr_msb 0x80c0
// LOCAL-DISASM-NEXT: ds_load_b64 v[0:1]{{.*v\[768:769\].*}}
// LOCAL-DISASM-NEXT: s_set_vgpr_msb 0xc080
// LOCAL-DISASM-NEXT: s_wait_dscnt 0x0

// RUN: hotswap-rewrite %t.local.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// A direct branch to the DS2 bypasses the setter. The local proof must reject
// that site even though DirectControlFlow remains closed.
// RUN: sed 's/^\.set bypass_setter, 0$/.set bypass_setter, 1/' \
// RUN:   %s > %t.bypass.s
// RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %t.bypass.s \
// RUN:   -o %t.bypass.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.bypass.o -o %t.bypass.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.bypass.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.bypass.out.elf --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=BYPASS-LOG,BYPASS-API %s
// RUN: test ! -e %t.bypass.out.elf
// BYPASS-LOG-NOT: unresolved call target
// BYPASS-LOG-NOT: unresolved control-flow target
// BYPASS-LOG: hotswap: error: ds_2addr at 0x{{[0-9A-F]+}}
// BYPASS-LOG-SAME: crosses v255 but the active VGPR-MSB mode is unknown
// BYPASS-API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.set bypass_setter, 0
.text

.globl test_ds_vgpr_msb_local_closed
.p2align 8
.type test_ds_vgpr_msb_local_closed,@function
test_ds_vgpr_msb_local_closed:
.if bypass_setter
  s_cbranch_scc1 .Lds2
.endif
  s_set_vgpr_msb 0x480
.Lds2:
  ds_load_2addr_b64 v[254:257], v32 offset0:64 offset1:96
  s_wait_dscnt 0x0
  s_endpgm

// Unreachable compiler-style reusable-PC tail. DirectControlFlow proves the
// exact local target, but VGPR-MSB whole-function recovery does not model this
// carry-chain materialization and leaves the function unanalyzed.
.Lgetpc:
  s_get_pc_i64 s[70:71]
  s_add_co_i32 s72, .Ltail_target-(.Lgetpc+4)-4, 4
  s_add_co_u32 s70, s70, s72
  s_add_co_ci_u32 s71, s71, 0
  s_set_pc_i64 s[70:71]
.Ltail_target:
  s_endpgm
.Ltest_ds_vgpr_msb_local_closed_end:
.size test_ds_vgpr_msb_local_closed, .Ltest_ds_vgpr_msb_local_closed_end-test_ds_vgpr_msb_local_closed

.rodata
.p2align 8
.amdhsa_kernel test_ds_vgpr_msb_local_closed
  .amdhsa_next_free_vgpr 770
  .amdhsa_next_free_sgpr 76
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds_vgpr_msb_local_closed
      .symbol: test_ds_vgpr_msb_local_closed.kd
      .sgpr_count: 76
      .vgpr_count: 770
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
