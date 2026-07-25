// COM: Exercise the entry-prefix displacement call site on a linked code
// COM: object with .eh_frame. The FDE must grow with the kernel instead of
// COM: forcing the appended-stub fallback. A renamed .eh_frame_hdr remains
// COM: unsupported because its binary-search table would need rebuilding.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: %llvm-readelf --section-headers %t.elf | \
// RUN:   %FileCheck --check-prefix=INPUT-SECTIONS %s
// INPUT-SECTIONS: .eh_frame
// RUN: %llvm-objdump --dwarf=frames %t.elf | \
// RUN:   %FileCheck --check-prefix=INPUT-FRAME %s
// INPUT-FRAME: FDE cie={{.*}} pc=00001500...0000150c

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --dump %t.out.elf --check-idempotent 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: displacement: remapped 1 .eh_frame FDE(s)
// API: hotswap: displacement: grew ELF
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | \
// RUN:   %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <eh_frame_kernel>:
// DISASM-NEXT: global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_setreg_imm32_b32
// DISASM-NEXT: s_endpgm

// RUN: %llvm-objdump --dwarf=frames %t.out.elf | \
// RUN:   %FileCheck --check-prefix=FRAME %s
// FRAME: .eh_frame contents:
// FRAME: CIE
// FRAME: Augmentation:          "zR"
// FRAME: FDE cie={{.*}} pc=00001500...0000151c
// FRAME-NOT: invalid

// RUN: %llvm-objcopy --rename-section .eh_frame=.eh_frame_hdr \
// RUN:   %t.elf %t.hdr.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.hdr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.hdr.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=HDR-API %s
// HDR-API: debug/unwind section '.eh_frame_hdr' requires address remapping
// HDR-API: using appended entry stubs
// HDR-API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.hdr.out.elf | \
// RUN:   %FileCheck --check-prefix=HDR-DISASM %s
// HDR-DISASM-LABEL: <eh_frame_kernel>:
// HDR-DISASM-NEXT: s_setreg_imm32_b32
// HDR-DISASM-NEXT: s_endpgm
// HDR-DISASM: global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
// HDR-DISASM-NEXT: v_nop
// HDR-DISASM-NEXT: s_get_pc_i64

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.cfi_sections .eh_frame
.text
.globl eh_frame_kernel
.p2align 8
.type eh_frame_kernel,@function
eh_frame_kernel:
  .cfi_startproc
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  s_endpgm
  .cfi_endproc
.Leh_frame_kernel_end:
.size eh_frame_kernel, .Leh_frame_kernel_end-eh_frame_kernel

.rodata
.p2align 8
.amdhsa_kernel eh_frame_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: eh_frame_kernel
      .symbol: eh_frame_kernel.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
