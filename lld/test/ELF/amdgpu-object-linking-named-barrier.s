# REQUIRES: amdgpu
## Test the assignment of named-barrier IDs. A barrier symbol of size 16 * N
## gets N consecutive IDs, and the first ID is 1. The linker uses the same
## frontier layout as for LDS. The symbol value encodes the first ID as
## 0x802000 | (ID << 4). The linker also writes the barrier count of each kernel
## to the kernel descriptor. All functions use flag 16 (FUNC_WAVE32), because
## gfx1250 does not support wave64.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=amdgpu12.50-amd-amdhsa a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu12.50-amd-amdhsa b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu12.50-amd-amdhsa c.s -o c.o
# RUN: ld.lld -shared a.o b.o c.o -o out.so
# RUN: llvm-readelf -s out.so | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -D -j .rodata out.so | FileCheck %s --check-prefix=KD

## kernel_a uses bar_a (1 ID) and calls helper in b.o, which uses bar_b
## (5 IDs). kernel_b uses bar_b. bar_b has more users, so it gets IDs 1-5, and
## bar_a gets ID 6.
## In c.o, each barrier has 2 IDs. K1 uses A and C, K2 uses A and B, and K3 uses
## B and D. C can use the IDs of B, because no kernel uses both B and C.
# SYM-DAG: 0000000000802060 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.bar_a{{$}}
# SYM-DAG: 0000000000802010 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.bar_b{{$}}
# SYM-DAG: 0000000000802010 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.A{{$}}
# SYM-DAG: 0000000000802030 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.B{{$}}
# SYM-DAG: 0000000000802030 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.C{{$}}
# SYM-DAG: 0000000000802050 0 NOTYPE LOCAL HIDDEN ABS __amdgpu_named_barrier.D{{$}}

## The count is the highest ID of the kernel, divided by 4 and rounded up.
# KD-LABEL: .amdhsa_kernel kernel_a
# KD:         .amdhsa_named_barrier_count 2
# KD-LABEL: .amdhsa_kernel kernel_b
# KD:         .amdhsa_named_barrier_count 2
# KD-LABEL: .amdhsa_kernel K1
# KD:         .amdhsa_named_barrier_count 1
# KD-LABEL: .amdhsa_kernel K2
# KD:         .amdhsa_named_barrier_count 1
# KD-LABEL: .amdhsa_kernel K3
# KD:         .amdhsa_named_barrier_count 2

## The highest allowed ID is 31.
# RUN: llvm-mc -filetype=obj -triple=amdgpu12.50-amd-amdhsa overflow.s -o overflow.o
# RUN: not ld.lld -shared overflow.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERFLOW --implicit-check-not=error:
# OVERFLOW: error: AMDGPU: named barrier ID overflow (max ID 32 exceeds limit of 31) in kernel 'kernel'

#--- kernel.inc
.macro kernel name
  .text
  .globl \name
  .p2align 8
\name:
  s_endpgm
  .rodata
  .p2align 6
  .globl \name\().kd
  .type \name\().kd, @object
  .size \name\().kd, 64
\name\().kd:
  .zero 64
.endm

#--- a.s
.include "kernel.inc"
  kernel kernel_a

  .amdgpu_info kernel_a
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.bar_a
    .amdgpu_call helper
  .end_amdgpu_info

  .globl __amdgpu_named_barrier.bar_a
  .amdgpu_lds __amdgpu_named_barrier.bar_a, 16, 4

#--- b.s
.include "kernel.inc"
  kernel kernel_b

  .text
  .globl helper
helper:
  s_setpc_b64 s[30:31]

  .amdgpu_info helper
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.bar_b
  .end_amdgpu_info
  .amdgpu_info kernel_b
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.bar_b
  .end_amdgpu_info

  .globl __amdgpu_named_barrier.bar_b
  .amdgpu_lds __amdgpu_named_barrier.bar_b, 80, 4

#--- c.s
.include "kernel.inc"
  kernel K1
  kernel K2
  kernel K3

  .amdgpu_info K1
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.A
    .amdgpu_use __amdgpu_named_barrier.C
  .end_amdgpu_info
  .amdgpu_info K2
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.A
    .amdgpu_use __amdgpu_named_barrier.B
  .end_amdgpu_info
  .amdgpu_info K3
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.B
    .amdgpu_use __amdgpu_named_barrier.D
  .end_amdgpu_info

  .globl __amdgpu_named_barrier.A, __amdgpu_named_barrier.B
  .globl __amdgpu_named_barrier.C, __amdgpu_named_barrier.D
  .amdgpu_lds __amdgpu_named_barrier.A, 32, 4
  .amdgpu_lds __amdgpu_named_barrier.B, 32, 4
  .amdgpu_lds __amdgpu_named_barrier.C, 32, 4
  .amdgpu_lds __amdgpu_named_barrier.D, 32, 4

#--- overflow.s
.include "kernel.inc"
  kernel kernel

  .amdgpu_info kernel
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_use __amdgpu_named_barrier.big
  .end_amdgpu_info

  .globl __amdgpu_named_barrier.big
  .amdgpu_lds __amdgpu_named_barrier.big, 512, 4
