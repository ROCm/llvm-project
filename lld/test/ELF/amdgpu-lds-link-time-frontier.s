# REQUIRES: amdgpu
## Test the LDS layout that uses the .amdgpu_use records. Each LDS symbol gets
## one offset for all kernels. Symbols with more kernel users come first. Ties
## are broken by alignment, then size, then name. A symbol starts at the highest
## frontier of its users, so two symbols without a common user can overlap.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa b.s -o b.o
# RUN: ld.lld -shared a.o b.o -o out.so
# RUN: llvm-readelf -s out.so | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -D -j .rodata out.so | FileCheck %s --check-prefix=KD

## Users: P {K1,K4,K5}, L {K2,K3}, M {K1,K2}, B {K5}, S {K5}, X {K4}, Y {K4},
## D1 {K1}, D2 {K3}, U {}. K3 uses L and D2 through g in b.o.
## - P has the most users.
## - L and M have 2 users each. L has the larger alignment, so it comes next. L
##   starts at 0 because no user of L uses P.
## - M starts at the frontier of K2 (16), which is higher than that of K1 (8).
## - B and S have the same alignment, so the larger B comes first.
## - X and Y have the same alignment and size, so they are sorted by name.
## - The size-0 (dynamic) symbols D1 and D2 start at the frontier of their
##   users, aligned to their own alignment. They do not move the frontier.
## - U has no kernel user. It still gets an offset, so that relocations to it
##   resolve, and it does not change the size of any kernel.
# SYM-DAG: 0000000000000000  8 NOTYPE LOCAL HIDDEN ABS P{{$}}
# SYM-DAG: 0000000000000000 16 NOTYPE LOCAL HIDDEN ABS L{{$}}
# SYM-DAG: 0000000000000010  4 NOTYPE LOCAL HIDDEN ABS M{{$}}
# SYM-DAG: 0000000000000008 32 NOTYPE LOCAL HIDDEN ABS B{{$}}
# SYM-DAG: 0000000000000028  8 NOTYPE LOCAL HIDDEN ABS S{{$}}
# SYM-DAG: 0000000000000008  4 NOTYPE LOCAL HIDDEN ABS X{{$}}
# SYM-DAG: 000000000000000c  4 NOTYPE LOCAL HIDDEN ABS Y{{$}}
# SYM-DAG: 0000000000000020  0 NOTYPE LOCAL HIDDEN ABS D1{{$}}
# SYM-DAG: 0000000000000010  0 NOTYPE LOCAL HIDDEN ABS D2{{$}}
# SYM-DAG: 0000000000000000 12 NOTYPE LOCAL HIDDEN ABS U{{$}}

## The group segment size of a kernel is the end of its last LDS symbol. For K1,
## this is the start of the dynamic symbol D1.
# KD-LABEL: .amdhsa_kernel K1
# KD-NEXT:    .amdhsa_group_segment_fixed_size 32
# KD-LABEL: .amdhsa_kernel K2
# KD-NEXT:    .amdhsa_group_segment_fixed_size 20
# KD-LABEL: .amdhsa_kernel K3
# KD-NEXT:    .amdhsa_group_segment_fixed_size 16
# KD-LABEL: .amdhsa_kernel K4
# KD-NEXT:    .amdhsa_group_segment_fixed_size 16
# KD-LABEL: .amdhsa_kernel K5
# KD-NEXT:    .amdhsa_group_segment_fixed_size 48

#--- a.s
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

  kernel K1
  kernel K2
  kernel K3
  kernel K4
  kernel K5

  .amdgpu_info K1
    .amdgpu_occupancy 4
    .amdgpu_use P
    .amdgpu_use M
    .amdgpu_use D1
  .end_amdgpu_info
  .amdgpu_info K2
    .amdgpu_occupancy 4
    .amdgpu_use M
    .amdgpu_use L
  .end_amdgpu_info
  .amdgpu_info K3
    .amdgpu_occupancy 4
    .amdgpu_call g
  .end_amdgpu_info
  .amdgpu_info K4
    .amdgpu_occupancy 4
    .amdgpu_use P
    .amdgpu_use Y
    .amdgpu_use X
  .end_amdgpu_info
  .amdgpu_info K5
    .amdgpu_occupancy 4
    .amdgpu_use P
    .amdgpu_use S
    .amdgpu_use B
  .end_amdgpu_info

  .globl P, M, B, S, X, Y, D1
  .amdgpu_lds P, 8, 8
  .amdgpu_lds M, 4, 4
  .amdgpu_lds B, 32, 4
  .amdgpu_lds S, 8, 4
  .amdgpu_lds X, 4, 4
  .amdgpu_lds Y, 4, 4
  .amdgpu_lds D1, 0, 16

#--- b.s
  .text
  .globl g, h
g:
  s_setpc_b64 s[30:31]
h:
  s_setpc_b64 s[30:31]

  .amdgpu_info g
    .amdgpu_occupancy 4
    .amdgpu_use L
    .amdgpu_use D2
  .end_amdgpu_info
  .amdgpu_info h
    .amdgpu_occupancy 4
    .amdgpu_use U
  .end_amdgpu_info

  .globl L, D2, U
  .amdgpu_lds L, 16, 16
  .amdgpu_lds D2, 0, 4
  .amdgpu_lds U, 12, 4
