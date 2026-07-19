// COM: A sized non-callable .text symbol protects its entire half-open extent
// COM: from use as donor storage, even when its point boundary precedes the
// COM: zero-filled gateway. An extent ending exactly at the gateway does not
// COM: overlap it.

// RUN: %clang -x assembler-with-cpp -DEXTENT_KIND=0 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.object.elf
// RUN: %llvm-readelf -s %t.object.elf \
// RUN:   | %FileCheck --check-prefix=OBJECT-SYM %s
// OBJECT-SYM: OBJECT LOCAL DEFAULT {{[0-9]+}} protected_extent
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.object.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=OVERLAP %s

// RUN: %clang -x assembler-with-cpp -DEXTENT_KIND=1 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.notype.elf
// RUN: %llvm-readelf -s %t.notype.elf \
// RUN:   | %FileCheck --check-prefix=NOTYPE-SYM %s
// NOTYPE-SYM: NOTYPE LOCAL DEFAULT {{[0-9]+}} protected_extent
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.notype.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck --check-prefix=OVERLAP %s

// RUN: %clang -x assembler-with-cpp -DEXTENT_KIND=2 \
// RUN:   -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.half-open.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.half-open.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.half-open.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=HALF-OPEN %s

// OVERLAP: hotswap: error: no safe short-branch gateway for far site
// OVERLAP: RESULT: ERROR
// HALF-OPEN: hotswap: assigned 1 SCC-neutral forward gateway(s)
// HALF-OPEN: RESULT: SUCCESS

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl gateway_kernel
.p2align 8
.type gateway_kernel,@function
gateway_kernel:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0
  s_endpgm
.size gateway_kernel, .-gateway_kernel

.local protected_extent
#if EXTENT_KIND != 1
.type protected_extent,@object
#endif
protected_extent:
  s_endpgm
#if EXTENT_KIND == 2
.size protected_extent, .-protected_extent
#endif
  .zero 32
#if EXTENT_KIND != 2
.size protected_extent, .-protected_extent
#endif

// Push the appended trampoline pool beyond direct s_branch reach.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel gateway_kernel
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: gateway_kernel
      .symbol: gateway_kernel.kd
      .sgpr_count: 66
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
