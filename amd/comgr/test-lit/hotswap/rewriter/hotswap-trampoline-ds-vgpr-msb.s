// Verify that splitting a two-address DS load whose second destination crosses
// the active VGPR bank rebases that half to v0 under a temporary destination
// VGPR-MSB mode without losing the nonzero source or destination bank.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: error:
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <test_ds_vgpr_msb>:
// DISASM:      s_set_vgpr_msb 0x41
// DISASM-NEXT: s_branch
// DISASM:      ds_load_b64 v[254:255], v88 offset:680
// DISASM-NEXT: s_set_vgpr_msb 0x4181
// DISASM-NEXT: ds_load_b64 v[0:1]{{.*v\[512:513\].*}}v88{{.*v344.*}}offset:688
// DISASM-NEXT: s_set_vgpr_msb 0x8141
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM:      s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_vgpr_msb
.p2align 8
.type test_ds_vgpr_msb,@function
test_ds_vgpr_msb:
  s_set_vgpr_msb 0x41
  ds_load_2addr_b64 v[254:257], v88 offset0:85 offset1:86
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds_vgpr_msb_end:
.size test_ds_vgpr_msb, .Ltest_ds_vgpr_msb_end-test_ds_vgpr_msb

.rodata
.p2align 8
.amdhsa_kernel test_ds_vgpr_msb
  .amdhsa_next_free_vgpr 688
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds_vgpr_msb
      .symbol: test_ds_vgpr_msb.kd
      .sgpr_count: 2
      .vgpr_count: 688
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
