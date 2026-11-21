# REQUIRES: amdgpu
## Test the propagation of resource usage through the call graph, and the
## patching of the kernel descriptors and the HSA metadata.

# RUN: rm -rf %t && split-file %s %t && cd %t

## kernel calls f and g in another object. The register counts use the
## maximum, the scratch size is the size of kernel plus that of its largest
## callee, and the flags are ORed. On gfx900 with xnack "any", flat scratch
## needs 6 extra SGPRs. kernel also uses LDS, so the linker patches the group
## segment size too.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa b.s -o b.o
# RUN: ld.lld -shared a.o b.o -o a.so
# RUN: llvm-objdump -D -j .rodata a.so | FileCheck %s --check-prefix=KD
# RUN: llvm-readobj --notes a.so | FileCheck %s --check-prefix=META
# KD:      .amdhsa_kernel kernel
# KD-NEXT:   .amdhsa_group_segment_fixed_size 256
# KD-NEXT:   .amdhsa_private_segment_fixed_size 116
# KD:        .amdhsa_next_free_vgpr 32
# KD:        .amdhsa_system_sgpr_private_segment_wavefront_offset 1
# KD:        .amdhsa_uses_dynamic_stack 1

# META:      - .agpr_count: 0
# META-NEXT:   .group_segment_fixed_size: 256
# META:        .name: kernel
# META-NEXT:   .private_segment_fixed_size: 116
# META-NEXT:   .sgpr_count: 46
# META-NEXT:   .symbol: kernel.kd
# META-NEXT:   .uses_dynamic_stack: true
# META-NEXT:   .vgpr_count: 30

## kernel uses 10 VGPRs and calls mfma, which uses 4 AGPRs. On gfx90a, the
## AGPRs follow the VGPRs aligned to 4, so the total is 12 + 4 = 16, and the
## accumulator offset is 12.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.0a-amd-amdhsa gfx90a.s -o gfx90a.o
# RUN: ld.lld -shared gfx90a.o -o gfx90a.so
# RUN: llvm-objdump -D -j .rodata gfx90a.so | FileCheck %s --check-prefix=GFX90A
# GFX90A: .amdhsa_accum_offset 12
# GFX90A: .amdhsa_next_free_vgpr 16

## kernel_h -> H -> A -> M -> H is a cycle, and A also calls C. kernel_m enters
## the cycle at M. Both kernels get the scratch size of the cycle (16) plus that
## of C (64), the VGPR count of C, the LDS of A, and a dynamic stack.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa recursion.s -o recursion.o
# RUN: ld.lld -shared recursion.o -o recursion.so
# RUN: llvm-objdump -D -j .rodata recursion.so | FileCheck %s --check-prefix=REC
# REC-LABEL: .amdhsa_kernel kernel_h
# REC-NEXT:    .amdhsa_group_segment_fixed_size 64
# REC-NEXT:    .amdhsa_private_segment_fixed_size 80
# REC:         .amdhsa_next_free_vgpr 60
# REC:         .amdhsa_uses_dynamic_stack 1
# REC-LABEL: .amdhsa_kernel kernel_m
# REC-NEXT:    .amdhsa_group_segment_fixed_size 64
# REC-NEXT:    .amdhsa_private_segment_fixed_size 80
# REC:         .amdhsa_next_free_vgpr 60
# REC:         .amdhsa_uses_dynamic_stack 1

## kernel calls _ZN3FooC1Ev, an alias of _ZN3FooC2Ev. Only _ZN3FooC2Ev has a
## .amdgpu.info record, and the alias uses it.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa alias.s -o alias.o
# RUN: ld.lld -shared alias.o -o alias.so
# RUN: llvm-objdump -D -j .rodata alias.so | FileCheck %s --check-prefix=ALIAS
# ALIAS: .amdhsa_next_free_vgpr 48

## callee is a local symbol in its own section, so the relocations in
## .amdgpu.info refer to the section symbol. The linker maps them back to
## callee.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa secsym.s -o secsym.o
# RUN: llvm-readobj -r secsym.o | FileCheck %s --check-prefix=SECSYM-REL
# RUN: ld.lld -shared secsym.o -o secsym.so
# RUN: llvm-objdump -D -j .rodata secsym.so | FileCheck %s --check-prefix=SECSYM
# SECSYM-REL: R_AMDGPU_ABS64 .text.unlikely. 0x0
# SECSYM:     .amdhsa_private_segment_fixed_size 132

## The .amdgpu.info records of the weak and COMDAT definitions that lose symbol
## resolution (in lose.o) are ignored.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa stale.s -o stale.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa win.s -o win.o
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa lose.s -o lose.o
# RUN: ld.lld -shared stale.o win.o lose.o -o stale.so
# RUN: llvm-objdump -D -j .rodata stale.so | FileCheck %s --check-prefix=STALE
# STALE:      .amdhsa_kernel kernel
# STALE-NEXT:   .amdhsa_group_segment_fixed_size 0
# STALE-NEXT:   .amdhsa_private_segment_fixed_size 32
# STALE:        .amdhsa_next_free_vgpr 12

## Each called function needs resource usage metadata.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa missing.s -o missing.o
# RUN: not ld.lld -shared missing.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING --implicit-check-not=error:
# MISSING-DAG: error: AMDGPU: function 'undefined_func' is undefined and has no resource usage metadata; called from 'kernel'
# MISSING-DAG: error: AMDGPU: function 'no_info' has no .amdgpu.info resource usage metadata; called from 'kernel'

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

  .amdgpu_info kernel
    .amdgpu_flags 3
    .amdgpu_num_vgpr 10
    .amdgpu_num_sgpr 20
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
    .amdgpu_use lds
    .amdgpu_call f
    .amdgpu_call g
  .end_amdgpu_info

  .globl lds
  .amdgpu_lds lds, 256, 16

  .amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: kernel.kd
    .vgpr_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
...
  .end_amdgpu_metadata

#--- b.s
  .text
  .globl f, g
f:
  s_setpc_b64 s[30:31]
g:
  s_setpc_b64 s[30:31]

  .amdgpu_info f
    .amdgpu_num_vgpr 30
    .amdgpu_num_sgpr 40
    .amdgpu_private_segment_size 64
    .amdgpu_occupancy 4
  .end_amdgpu_info
;; Flag 4 is FUNC_HAS_DYN_STACK.
  .amdgpu_info g
    .amdgpu_flags 4
    .amdgpu_num_vgpr 5
    .amdgpu_num_sgpr 2
    .amdgpu_private_segment_size 100
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- gfx90a.s
.include "kernel.inc"
  kernel kernel

  .text
  .globl mfma
mfma:
  s_setpc_b64 s[30:31]

  .amdgpu_info kernel
    .amdgpu_num_vgpr 10
    .amdgpu_occupancy 4
    .amdgpu_call mfma
  .end_amdgpu_info
  .amdgpu_info mfma
    .amdgpu_num_vgpr 6
    .amdgpu_num_agpr 4
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- recursion.s
.include "kernel.inc"
  kernel kernel_h
  kernel kernel_m

  .text
  .globl H, A, M, C
H:
  s_setpc_b64 s[30:31]
A:
  s_setpc_b64 s[30:31]
M:
  s_setpc_b64 s[30:31]
C:
  s_setpc_b64 s[30:31]

  .amdgpu_info kernel_h
    .amdgpu_num_vgpr 10
    .amdgpu_occupancy 4
    .amdgpu_call H
  .end_amdgpu_info
  .amdgpu_info kernel_m
    .amdgpu_num_vgpr 10
    .amdgpu_occupancy 4
    .amdgpu_call M
  .end_amdgpu_info
  .amdgpu_info H
    .amdgpu_num_vgpr 10
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
    .amdgpu_call A
  .end_amdgpu_info
  .amdgpu_info A
    .amdgpu_num_vgpr 10
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
    .amdgpu_use lds
    .amdgpu_call M
    .amdgpu_call C
  .end_amdgpu_info
  .amdgpu_info M
    .amdgpu_num_vgpr 10
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
    .amdgpu_call H
  .end_amdgpu_info
  .amdgpu_info C
    .amdgpu_num_vgpr 60
    .amdgpu_private_segment_size 64
    .amdgpu_occupancy 4
  .end_amdgpu_info

  .globl lds
  .amdgpu_lds lds, 64, 16

#--- alias.s
.include "kernel.inc"
  kernel kernel

  .text
  .globl _ZN3FooC2Ev, _ZN3FooC1Ev
_ZN3FooC2Ev:
  s_setpc_b64 s[30:31]
  .set _ZN3FooC1Ev, _ZN3FooC2Ev

  .amdgpu_info kernel
    .amdgpu_occupancy 4
    .amdgpu_call _ZN3FooC1Ev
  .end_amdgpu_info
  .amdgpu_info _ZN3FooC2Ev
    .amdgpu_num_vgpr 48
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- secsym.s
.include "kernel.inc"
  kernel kernel

  .section .text.unlikely.,"ax",@progbits
callee:
  s_setpc_b64 s[30:31]

  .amdgpu_info kernel
    .amdgpu_occupancy 4
    .amdgpu_call callee
  .end_amdgpu_info
  .amdgpu_info callee
    .amdgpu_private_segment_size 132
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- stale.s
.include "kernel.inc"
  kernel kernel

  .amdgpu_info kernel
    .amdgpu_occupancy 4
    .amdgpu_call weak_helper
    .amdgpu_call comdat_helper
  .end_amdgpu_info

#--- win.s
  .text
  .weak weak_helper
weak_helper:
  s_setpc_b64 s[30:31]
  .section .text.comdat_helper,"axG",@progbits,comdat_helper,comdat
  .globl comdat_helper
comdat_helper:
  s_setpc_b64 s[30:31]

  .amdgpu_info weak_helper
    .amdgpu_num_vgpr 8
    .amdgpu_private_segment_size 16
    .amdgpu_occupancy 4
  .end_amdgpu_info
  .amdgpu_info comdat_helper
    .amdgpu_num_vgpr 12
    .amdgpu_private_segment_size 32
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- lose.s
  .text
  .weak weak_helper
weak_helper:
  s_setpc_b64 s[30:31]
  .section .text.comdat_helper,"axG",@progbits,comdat_helper,comdat
  .globl comdat_helper
comdat_helper:
  s_setpc_b64 s[30:31]

  .amdgpu_info weak_helper
    .amdgpu_num_vgpr 40
    .amdgpu_private_segment_size 128
    .amdgpu_occupancy 4
  .end_amdgpu_info
  .amdgpu_info comdat_helper
    .amdgpu_num_vgpr 44
    .amdgpu_private_segment_size 256
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- missing.s
.include "kernel.inc"
  kernel kernel

  .text
  .globl no_info
no_info:
  s_setpc_b64 s[30:31]

  .amdgpu_info kernel
    .amdgpu_occupancy 4
    .amdgpu_call undefined_func
    .amdgpu_call no_info
  .end_amdgpu_info
