// COM: A far patch can live in an ordinary device function shared by multiple
// COM: kernels. The sign-extended literal32 s_add_pc_i64 return uses no SGPR,
// COM: so it requires neither a device-function descriptor nor caller metadata
// COM: changes.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s

// DISASM-LABEL: <device_tensor>:
// DISASM-NEXT: s_add_pc_i64
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_add_pc_i64 0xffff{{[0-9a-f]+}}

// META: .name:           kernel_a
// META: .sgpr_count:     48
// META: .name:           kernel_b
// META: .sgpr_count:     48

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.local device_tensor
.type device_tensor,@function
device_tensor:
  tensor_load_to_lds s[0:3], s[4:11]
  s_set_pc_i64 s[30:31]
.Ldevice_tensor_end:
.size device_tensor, .Ldevice_tensor_end-device_tensor

.globl kernel_a
.type kernel_a,@function
kernel_a:
  s_call_i64 s[30:31], device_tensor
  s_endpgm
.Lkernel_a_end:
.size kernel_a, .Lkernel_a_end-kernel_a

.globl kernel_b
.type kernel_b,@function
kernel_b:
  s_call_i64 s[30:31], device_tensor
  s_endpgm
.Lkernel_b_end:
.size kernel_b, .Lkernel_b_end-kernel_b

// Push the appended trampoline pool beyond s_branch reach without extending
// any function symbol range.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel kernel_a
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 48
.end_amdhsa_kernel

.amdhsa_kernel kernel_b
  .amdhsa_next_free_vgpr 1
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
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: kernel_b
      .symbol: kernel_b.kd
      .sgpr_count: 48
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
