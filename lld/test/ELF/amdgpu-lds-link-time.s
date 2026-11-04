# REQUIRES: amdgpu
## Test the link-time resolution of SHN_AMDGPU_LDS symbols when no input has
## .amdgpu_use records. All LDS symbols then share one layout, sorted by
## alignment, then size, then name, and each kernel uses the whole layout. The
## symbols become absolute and hidden, and the linker resolves the
## R_AMDGPU_ABS32_LO relocations that refer to them.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa b.s -o b.o
# RUN: ld.lld -shared a.o b.o -o out.so
# RUN: llvm-readelf -s out.so | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -d out.so | FileCheck %s --check-prefix=DIS
# RUN: llvm-objdump -D -j .rodata out.so | FileCheck %s --check-prefix=KD

## lds_a16_s4 comes first, although it is the smallest symbol. lds_a8_s64 and
## lds_a4_y need padding. lds_a4_x and lds_a4_y are sorted by name.
# SYM-DAG: 0000000000000000  4 NOTYPE LOCAL HIDDEN ABS lds_a16_s4{{$}}
# SYM-DAG: 0000000000000008 64 NOTYPE LOCAL HIDDEN ABS lds_a8_s64{{$}}
# SYM-DAG: 0000000000000048 24 NOTYPE LOCAL HIDDEN ABS lds_a8_s24{{$}}
# SYM-DAG: 0000000000000060  6 NOTYPE LOCAL HIDDEN ABS lds_a4_x{{$}}
# SYM-DAG: 0000000000000068  6 NOTYPE LOCAL HIDDEN ABS lds_a4_y{{$}}
# SYM-DAG: 000000000000006e  1 NOTYPE LOCAL HIDDEN ABS lds_a1{{$}}

## The relocations refer to an LDS symbol of b.o and to one of a.o.
# DIS-LABEL: <kernel>:
# DIS-NEXT:    s_mov_b32 s0, lit(0x8)
# DIS-NEXT:    s_mov_b32 s1, 0x60

# KD:      .amdhsa_kernel kernel
# KD-NEXT:   .amdhsa_group_segment_fixed_size 111

## A link that has LDS symbols but no kernel cannot assign offsets.
# RUN: not ld.lld -shared b.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-KERNEL --implicit-check-not=error:
# NO-KERNEL: error: cannot resolve AMDGPU LDS symbols without a kernel descriptor and .amdgpu.info metadata

## LDS resolution requires the HSA OS ABI.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdpal pal.s -o pal.o
# RUN: not ld.lld -shared pal.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NON-HSA --implicit-check-not=error:
# NON-HSA: error: object linking for AMDGPU LDS requires the HSA OSABI

## The assembler rejects an alignment that is not a power of two, so use
## yaml2obj to make such a symbol.
# RUN: yaml2obj align.yaml -o align.o
# RUN: not ld.lld -shared align.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALIGN --implicit-check-not=error:
# ALIGN: error: symbol 'lds' has non-power-of-two AMDGPU LDS alignment 3

#--- a.s
  .text
  .globl kernel
  .p2align 8
kernel:
  s_mov_b32 s0, lds_a8_s64@abs32@lo
  s_mov_b32 s1, lds_a4_x@abs32@lo
  s_endpgm

  .rodata
  .p2align 6
  .globl kernel.kd
  .type kernel.kd, @object
  .size kernel.kd, 64
kernel.kd:
  .zero 64

  .amdgpu_info kernel
    .amdgpu_occupancy 4
  .end_amdgpu_info

  .globl lds_a16_s4, lds_a4_x, lds_a1
  .amdgpu_lds lds_a16_s4, 4, 16
  .amdgpu_lds lds_a4_x, 6, 4
  .amdgpu_lds lds_a1, 1, 1

#--- b.s
  .globl lds_a8_s64, lds_a8_s24, lds_a4_y
  .amdgpu_lds lds_a8_s64, 64, 8
  .amdgpu_lds lds_a8_s24, 24, 8
  .amdgpu_lds lds_a4_y, 6, 4

#--- pal.s
  .globl lds
  .amdgpu_lds lds, 4, 4

#--- align.yaml
--- !ELF
FileHeader:
  Class:      ELFCLASS64
  Data:       ELFDATA2LSB
  OSABI:      ELFOSABI_AMDGPU_HSA
  ABIVersion: 4
  Type:       ET_REL
  Machine:    EM_AMDGPU
  Flags:      [ EF_AMDGPU_MACH_AMDGCN_GFX900 ]
Symbols:
  - Name:    lds
    Index:   0xff00
    Value:   3
    Size:    4
    Binding: STB_GLOBAL
