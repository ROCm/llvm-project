// COM: A generic gfx1250 request uses uniform B0 kernel metadata as the
// COM: durable authorization for exact 8-byte DS2 offset rewrites. The first
// COM: pass retags that metadata to A0; repeating the identical generic request
// COM: must preserve every rewritten instruction byte-for-byte.
// COM:
// COM: The single kernel covers every non-stride DS2 dispatch entry:
// COM: b32/b64 load, store, and returning exchange.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: generic gfx1250 source has uniform B0 metadata
// LOG: hotswap: rewrote ds_load_2addr_b32
// LOG: hotswap: rewrote ds_load_2addr_b64
// LOG: hotswap: rewrote ds_store_2addr_b32
// LOG: hotswap: rewrote ds_store_2addr_b64
// LOG: hotswap: rewrote ds_storexchg_2addr_rtn_b32
// LOG: hotswap: rewrote ds_storexchg_2addr_rtn_b64
// API: RESULT: SUCCESS

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_ds2_generic_idempotent>:
// DISASM-NEXT: ds_load_2addr_b32 v[0:1], v24 offset0:4 offset1:8
// DISASM-NEXT: ds_load_2addr_b64 v[2:5], v24 offset0:8 offset1:16
// DISASM-NEXT: ds_store_2addr_b32 v24, v6, v7 offset0:12 offset1:16
// DISASM-NEXT: ds_store_2addr_b64 v24, v[8:9], v[10:11] offset0:8 offset1:16
// DISASM-NEXT: ds_storexchg_2addr_rtn_b32 v[12:13], v24, v14, v15 offset0:20 offset1:24
// DISASM-NEXT: ds_storexchg_2addr_rtn_b64 v[16:19], v24, v[20:21], v[22:23] offset0:8 offset1:16
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// METADATA: .gfx1250_revision: A0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2_generic_idempotent
.p2align 8
.type test_ds2_generic_idempotent,@function
test_ds2_generic_idempotent:
  ds_load_2addr_b32 v[0:1], v24 offset0:1 offset1:2
  ds_load_2addr_b64 v[2:5], v24 offset0:1 offset1:2
  ds_store_2addr_b32 v24, v6, v7 offset0:3 offset1:4
  ds_store_2addr_b64 v24, v[8:9], v[10:11] offset0:1 offset1:2
  ds_storexchg_2addr_rtn_b32 v[12:13], v24, v14, v15 offset0:5 offset1:6
  ds_storexchg_2addr_rtn_b64 v[16:19], v24, v[20:21], v[22:23] offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2_generic_idempotent_end:
.size test_ds2_generic_idempotent, .Ltest_ds2_generic_idempotent_end-test_ds2_generic_idempotent

.rodata
.p2align 8
.amdhsa_kernel test_ds2_generic_idempotent
  .amdhsa_next_free_vgpr 25
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_ds2_generic_idempotent
      .symbol: test_ds2_generic_idempotent.kd
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
