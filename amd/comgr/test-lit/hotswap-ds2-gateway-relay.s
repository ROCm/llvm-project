// COM: A far DS2 site has no safe SGPR/VCC scratch. Its replacement body
// COM: belongs in an in-function tail NOP sled. An unreachable zero-filled
// COM: hole midway through the function is usable only for the two branch
// COM: gateway dwords that relay between the source and body.

// RUN: %clang -x assembler-with-cpp -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=API %s
// API-NOT: hotswap: error:
// API: hotswap: DS2 gateway plan: used 1 entry gateway(s), 1 return gateway(s), and body sled at 0x20020 for site 0x0
// API-NOT: hotswap: error:
// API: hotswap: applied 1 instruction patches
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <ds2_gateway_relay>:
// DISASM-NEXT: s_branch {{.*}}<ds2_gateway_relay+0x10004>
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NOT: ds_load_2addr_b64
// DISASM: s_branch 49152
// DISASM-NEXT: s_branch {{.*}}<ds2_gateway_relay+0x20020>
// DISASM: ds_load_b64 v[0:1], v4 offset:8
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:16
// DISASM-NEXT: s_branch {{.*}}<ds2_gateway_relay+0x10000>
// DISASM-NOT: ds_load_2addr_b64
// DISASM: <metadata_anchor>:

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: REWRITE: SUCCESS
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local ds2_gateway_relay
.type ds2_gateway_relay,@function
ds2_gateway_relay:
  ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2

// Put a two-dword zero-filled gateway halfway to the body. The branch around
// it makes the hole unreachable by fallthrough, while its interior has no
// callable entry or direct branch target.
.rept (0x10000 - 8 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_branch .Lgateway_resume
.Lgateway_padding:
  .zero 8
.Lgateway_resume:

// The body starts just beyond direct source reach. It is a real NOP sled:
// inside the source function, after a no-fallthrough instruction, and ending
// exactly at the function boundary.
.rept (0x20020 - 0x10008 - 4) / 4
  s_mov_b32 s0, s0
.endr
  s_set_pc_i64 s[30:31]
.Lbody_sled:
.rept 5
  s_nop 0
.endr
.size ds2_gateway_relay, .-ds2_gateway_relay

.globl metadata_anchor
.type metadata_anchor,@function
metadata_anchor:
  s_endpgm
.size metadata_anchor, .-metadata_anchor

.rodata
.p2align 8
.amdhsa_kernel metadata_anchor
  .amdhsa_next_free_vgpr 17
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: metadata_anchor
      .symbol: metadata_anchor.kd
      .sgpr_count: 106
      .vgpr_count: 17
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
