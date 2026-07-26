// COM: Splitting this returning exchange would have cyclic destination/source
// COM: dependencies, but its scaled byte offsets fit the original DS2 fields.
// COM: Uniform B0 metadata therefore permits the exact 8-byte in-place rewrite,
// COM: preserving compound read-before-write semantics and generic idempotence.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: generic gfx1250 source has uniform B0 metadata
// LOG: hotswap: rewrote ds_storexchg_2addr_rtn_b64
// LOG-SAME: in place with A0 byte offsets 0, 8
// LOG-NOT: cyclic destination/source overlap
// API: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s
// DISASM-LABEL: <test_ds2_cyclic_inplace>:
// DISASM-NEXT: ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] offset1:8
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// METADATA: .gfx1250_revision: A0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_cyclic_inplace
.p2align 8
.type test_ds2_cyclic_inplace,@function
test_ds2_cyclic_inplace:
  ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] offset0:0 offset1:1
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2_cyclic_inplace_end:
.size test_ds2_cyclic_inplace, .Ltest_ds2_cyclic_inplace_end-test_ds2_cyclic_inplace

.rodata
.p2align 8
.amdhsa_kernel test_ds2_cyclic_inplace
  .amdhsa_next_free_vgpr 25
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_ds2_cyclic_inplace
      .symbol: test_ds2_cyclic_inplace.kd
      .gfx1250_revision: B0
      .sgpr_count: 1
      .vgpr_count: 25
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
