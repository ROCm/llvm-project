// COM: A far DS2 patch can live in an ordinary device function shared by
// COM: multiple kernels. The set-PC return uses one globally unused SGPR pair
// COM: and charges that pair to both possible callers.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s

// DISASM-LABEL: <device_ds2>:
// DISASM-NEXT: s_call_i64 s[66:67],
// DISASM-LABEL: <kernel_b>:
// DISASM: s_endpgm
// DISASM-NEXT: s_add_nc_u64 s[66:67], s[66:67],
// DISASM-NEXT: s_set_pc_i64 s[66:67]
// DISASM: ds_load_b64 v[0:1], v4 offset:512
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:1024
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64 s[66:67]
// DISASM-NEXT: s_add_nc_u64 s[66:67], s[66:67],
// DISASM-NEXT: s_set_pc_i64 s[66:67]

// META: .name:           kernel_a
// META: .sgpr_count:     70
// META: .name:           kernel_b
// META: .sgpr_count:     70

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.local device_ds2
.type device_ds2,@function
device_ds2:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_set_pc_i64 s[30:31]
.Ldevice_ds2_end:
.size device_ds2, .Ldevice_ds2_end-device_ds2

.globl kernel_a
.type kernel_a,@function
kernel_a:
  s_call_i64 s[30:31], device_ds2
  s_endpgm
.Lkernel_a_end:
.size kernel_a, .Lkernel_a_end-kernel_a

.globl kernel_b
.type kernel_b,@function
kernel_b:
  s_call_i64 s[30:31], device_ds2
  s_endpgm
.Lkernel_b_end:
.size kernel_b, .Lkernel_b_end-kernel_b

// Safe gateway space outside every function range.
.rept 8
  s_nop 0
.endr

// Push the appended pool beyond s_branch reach without extending a function.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel kernel_a
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 48
.end_amdhsa_kernel

.amdhsa_kernel kernel_b
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 48
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: kernel_a
      .symbol: kernel_a.kd
      .sgpr_count: 48
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: kernel_b
      .symbol: kernel_b.kd
      .sgpr_count: 48
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
