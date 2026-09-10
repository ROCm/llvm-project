// COM: Far eight-byte patch sites have no padding inside their tiny owning
// COM: functions. Their unreachable second dwords form a registerless relay
// COM: chain to the appended trampoline pool.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: assigned 1 forward s_branch island chain(s)
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <source0>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop
// DISASM-LABEL: <source1>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl source0
.p2align 8
.type source0,@function
source0:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size source0, .-source0

.rept 25000
  s_mov_b32 s0, s1
.endr

.globl source1
.type source1,@function
source1:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_endpgm
.size source1, .-source1

.rept 12500
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel source0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdhsa_kernel source1
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: source0
      .symbol: source0.kd
      .sgpr_count: 14
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: source1
      .symbol: source1.kd
      .sgpr_count: 14
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
