# REQUIRES: amdgpu
## Test indirect call edges. An indirect call with type ID T can call each
## address-taken function with type ID T. The type ID of a function can come
## from another object, which takes its address. LDS use and resource usage
## propagate through these edges.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa b.s -o b.o
# RUN: ld.lld -shared a.o b.o -o out.so
# RUN: llvm-readelf -s out.so | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -D -j .rodata out.so | FileCheck %s --check-prefix=KD

## kernel -> caller --"vi"--> t1, t2, t4. a.o takes the address of t4, which
## b.o defines. t3 has the type ID "iii", so no kernel uses C.
# SYM-DAG: 0000000000000000  64 NOTYPE LOCAL HIDDEN ABS A{{$}}
# SYM-DAG: 0000000000000040  32 NOTYPE LOCAL HIDDEN ABS B{{$}}
# SYM-DAG: 0000000000000060  16 NOTYPE LOCAL HIDDEN ABS D{{$}}
# SYM-DAG: 0000000000000000 128 NOTYPE LOCAL HIDDEN ABS C{{$}}

## The scratch size is that of kernel (0), plus caller (16), plus the largest
## indirect callee (t1, 32). The VGPR count comes from t1.
# KD:      .amdhsa_kernel kernel
# KD-NEXT:   .amdhsa_group_segment_fixed_size 112
# KD-NEXT:   .amdhsa_private_segment_fixed_size 48
# KD:        .amdhsa_next_free_vgpr 40

## The type ID at the end of .amdgpu.strtab has no NUL terminator. The linker
## must not read past the section, and must still match the indirect call.
## The record of the unknown kind 12 is skipped.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa strtab.s -o strtab.o
# RUN: ld.lld -shared strtab.o -o strtab.so
# RUN: llvm-objdump -D -j .rodata strtab.so | FileCheck %s --check-prefix=STRTAB
# STRTAB:      .amdhsa_kernel kernel
# STRTAB-NEXT:   .amdhsa_group_segment_fixed_size 128

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
  kernel kernel

  .text
  .globl caller, t1, t2, t3
caller:
  s_setpc_b64 s[30:31]
t1:
  s_setpc_b64 s[30:31]
t2:
  s_setpc_b64 s[30:31]
t3:
  s_setpc_b64 s[30:31]

  .amdgpu_info kernel
    .amdgpu_occupancy 4
    .amdgpu_call caller
  .end_amdgpu_info
  .amdgpu_info caller
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
    .amdgpu_indirect_call "vi"
  .end_amdgpu_info
  .amdgpu_info t1
    .amdgpu_num_vgpr 40
    .amdgpu_private_segment_size 32
    .amdgpu_occupancy 4
    .amdgpu_use A
    .amdgpu_typeid "vi"
  .end_amdgpu_info
  .amdgpu_info t2
    .amdgpu_occupancy 4
    .amdgpu_use B
    .amdgpu_typeid "vi"
  .end_amdgpu_info
  .amdgpu_info t3
    .amdgpu_occupancy 4
    .amdgpu_use C
    .amdgpu_typeid "iii"
  .end_amdgpu_info
  .amdgpu_info t4
    .amdgpu_typeid "vi"
  .end_amdgpu_info

  .globl A, B, C
  .amdgpu_lds A, 64, 16
  .amdgpu_lds B, 32, 16
  .amdgpu_lds C, 128, 16

#--- b.s
  .text
  .globl t4
t4:
  s_setpc_b64 s[30:31]

  .amdgpu_info t4
    .amdgpu_occupancy 4
    .amdgpu_use D
  .end_amdgpu_info

  .globl D
  .amdgpu_lds D, 16, 16

#--- strtab.s
.include "kernel.inc"
  kernel kernel

  .text
  .globl caller, target
caller:
  s_setpc_b64 s[30:31]
target:
  s_setpc_b64 s[30:31]

  .globl lds
  .amdgpu_lds lds, 128, 16

;; Each record is [kind: u8][size: u8][payload]. The kinds are 1 (function),
;; 2 (flags), 7 (use), 8 (call), 9 (indirect call), 10 (type ID), and
;; 11 (occupancy).
  .section .amdgpu.info,"e",@progbits
  .byte 1, 8
  .quad kernel
  .byte 2, 4
  .long 0
  .byte 11, 4
  .long 4
  .byte 12, 4
  .long 0
  .byte 8, 8
  .quad caller

  .byte 1, 8
  .quad caller
  .byte 2, 4
  .long 0
  .byte 11, 4
  .long 4
  .byte 9, 4
  .long 0

  .byte 1, 8
  .quad target
  .byte 2, 4
  .long 0
  .byte 11, 4
  .long 4
  .byte 7, 8
  .quad lds
  .byte 10, 4
  .long 0

  .section .amdgpu.strtab,"e",@progbits
  .ascii "vi"
