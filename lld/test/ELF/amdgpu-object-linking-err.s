# REQUIRES: amdgpu
## Test the errors of AMDGPU object linking for incompatible functions and
## input files. Flag 8 is FUNC_WGP_MODE and flag 16 is FUNC_WAVE32.

# RUN: rm -rf %t && split-file %s %t && cd %t

## A caller and its callee must use the same wavefront size and the same CU/WGP
## mode. This also applies to an indirect call.
# RUN: llvm-mc -filetype=obj -triple=amdgpu10.30-amd-amdhsa wave-size.s -o wave-size.o
# RUN: not ld.lld -shared wave-size.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WAVE-SIZE --implicit-check-not=error:
# WAVE-SIZE: error: AMDGPU object linking: wave size mismatch in call from 'kernel' (wave32) to 'helper' (wave64)

# RUN: llvm-mc -filetype=obj -triple=amdgpu10.30-amd-amdhsa mode-call.s -o mode-call.o
# RUN: not ld.lld -shared mode-call.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE-CALL --implicit-check-not=error:
# MODE-CALL: error: AMDGPU object linking: CU/WGP mode mismatch in call from 'kernel' (WGP) to 'helper' (CU)

# RUN: llvm-mc -filetype=obj -triple=amdgpu10.30-amd-amdhsa mode-icall.s -o mode-icall.o
# RUN: not ld.lld -shared mode-icall.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE-ICALL --implicit-check-not=error:
# MODE-ICALL: error: AMDGPU object linking: CU/WGP mode mismatch in call from 'kernel' (WGP) to 'helper' (CU)

## The target must support the wavefront size.
# RUN: llvm-mc -filetype=obj -triple=amdgpu12.50-amd-amdhsa wave64.s -o wave64.o
# RUN: not ld.lld -shared wave64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WAVE64 --implicit-check-not=error:
# WAVE64: error: AMDGPU: function 'kernel' uses unsupported ABI wave64 for gfx1250

# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa wave32.s -o wave32.o
# RUN: not ld.lld -shared wave32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WAVE32 --implicit-check-not=error:
# WAVE32: error: AMDGPU: function 'kernel' uses unsupported ABI wave32 for gfx900

## A function record without flags has no wavefront size. The assembler always
## writes the flags, so use raw records.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa no-flags.s -o no-flags.o
# RUN: not ld.lld -shared no-flags.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-FLAGS --implicit-check-not=error:
# NO-FLAGS: error: AMDGPU: function 'kernel' has invalid ABI wave size 0

## Each function needs an ABI occupancy, and a callee must support at least the
## occupancy of its caller.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa occ-zero.s -o occ-zero.o
# RUN: not ld.lld -shared occ-zero.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OCC-ZERO --implicit-check-not=error:
# OCC-ZERO: error: AMDGPU: function 'kernel' has invalid ABI occupancy 0

# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa occ-call.s -o occ-call.o
# RUN: not ld.lld -shared occ-call.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OCC-CALL --implicit-check-not=error:
# OCC-CALL: error: AMDGPU: incompatible ABI occupancy: function 'kernel' requires occupancy 8 but calls 'helper' with occupancy 4

## The LDS of a kernel must fit its ABI occupancy. With 10 waves per EU, 3
## workgroups of 1024 work-items must fit in the 64 KiB of a gfx900 CU.
# RUN: llvm-mc -filetype=obj -triple=amdgpu9.00-amd-amdhsa lds-occ.s -o lds-occ.o
# RUN: not ld.lld -shared lds-occ.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=LDS-OCC --implicit-check-not=error:
# LDS-OCC: error: AMDGPU: kernel 'kernel' uses 65536 bytes of LDS, which does not meet ABI occupancy 10

# RUN: yaml2obj big-endian.yaml -o big-endian.o
# RUN: not ld.lld -shared big-endian.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BIG-ENDIAN --implicit-check-not=error:
# BIG-ENDIAN: error: AMDGPU only supports little-endian ELF

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

.macro helper
  .text
  .globl helper
helper:
  s_setpc_b64 s[30:31]
.endm

#--- wave-size.s
.include "kernel.inc"
  kernel kernel
  helper
  .amdgpu_info kernel
    .amdgpu_flags 16
    .amdgpu_occupancy 8
    .amdgpu_call helper
  .end_amdgpu_info
  .amdgpu_info helper
    .amdgpu_occupancy 8
  .end_amdgpu_info

#--- mode-call.s
.include "kernel.inc"
  kernel kernel
  helper
  .amdgpu_info kernel
    .amdgpu_flags 8
    .amdgpu_occupancy 4
    .amdgpu_call helper
  .end_amdgpu_info
  .amdgpu_info helper
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- mode-icall.s
.include "kernel.inc"
  kernel kernel
  helper
  .amdgpu_info kernel
    .amdgpu_flags 8
    .amdgpu_occupancy 4
    .amdgpu_indirect_call "v"
  .end_amdgpu_info
  .amdgpu_info helper
    .amdgpu_occupancy 4
    .amdgpu_typeid "v"
  .end_amdgpu_info

#--- wave64.s
.include "kernel.inc"
  kernel kernel
  .amdgpu_info kernel
    .amdgpu_occupancy 8
  .end_amdgpu_info

#--- wave32.s
.include "kernel.inc"
  kernel kernel
  .amdgpu_info kernel
    .amdgpu_flags 16
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- no-flags.s
.include "kernel.inc"
  kernel kernel
;; A function record (kind 1) with only an occupancy record (kind 11).
  .section .amdgpu.info,"e",@progbits
  .byte 1, 8
  .quad kernel
  .byte 11, 4
  .long 4

#--- occ-zero.s
.include "kernel.inc"
  kernel kernel
  .amdgpu_info kernel
    .amdgpu_flags 0
  .end_amdgpu_info

#--- occ-call.s
.include "kernel.inc"
  kernel kernel
  helper
  .amdgpu_info kernel
    .amdgpu_occupancy 8
    .amdgpu_call helper
  .end_amdgpu_info
  .amdgpu_info helper
    .amdgpu_occupancy 4
  .end_amdgpu_info

#--- lds-occ.s
.include "kernel.inc"
  kernel kernel
  .amdgpu_info kernel
    .amdgpu_occupancy 10
    .amdgpu_use lds
  .end_amdgpu_info
  .globl lds
  .amdgpu_lds lds, 65536, 16

#--- big-endian.yaml
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2MSB
  Type:    ET_REL
  Machine: EM_AMDGPU
Sections:
  - Name:  .text
    Type:  SHT_PROGBITS
    Flags: [ SHF_ALLOC, SHF_EXECINSTR ]
    Size:  4
