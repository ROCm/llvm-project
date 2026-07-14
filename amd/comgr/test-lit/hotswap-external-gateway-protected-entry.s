// COM: Zero-filled external padding cannot donate a far-branch gateway when it
// COM: contains any emitted symbol, a decoded direct target, or a
// COM: kernel-descriptor entry. Only a temporary .L assembly anchor omitted
// COM: from the final symbol table leaves the otherwise unreachable run usable.

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=0 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.control.elf
// RUN: %llvm-readelf -s %t.control.elf \
// RUN:   | %FileCheck --check-prefix=CONTROL-SYMS %s
// CONTROL-SYMS: Symbol table
// CONTROL-SYMS-NOT: protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.control.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.control.out.elf 2>&1 | %FileCheck --check-prefix=CONTROL %s
// RUN: %llvm-objdump -d %t.control.out.elf \
// RUN:   | %FileCheck --check-prefix=CONTROL-DISASM %s
// CONTROL: hotswap: assigned 1 SCC-neutral forward gateway(s)
// CONTROL: RESULT: SUCCESS
// CONTROL-DISASM: s_get_pc_i64

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=1 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.func.elf
// RUN: %llvm-readelf -s %t.func.elf | %FileCheck --check-prefix=FUNC-SYM %s
// FUNC-SYM: FUNC LOCAL DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.func.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// COM: STT_GNU_IFUNC and STT_AMDGPU_HSA_KERNEL both have value 10, so LLVM's
// COM: AMDGPU readelf spelling for this symbol is target-specific.
// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=2 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.ifunc.elf
// RUN: %llvm-readelf -s %t.ifunc.elf | %FileCheck --check-prefix=IFUNC-SYM %s
// IFUNC-SYM: AMDGPU_HSA_KERNEL LOCAL DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.ifunc.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=3 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.kd.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.kd.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=4 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.object.elf
// RUN: %llvm-readelf -s %t.object.elf | %FileCheck --check-prefix=OBJECT-SYM %s
// OBJECT-SYM: OBJECT LOCAL DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.object.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=5 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.global.elf
// RUN: %llvm-readelf -s %t.global.elf | %FileCheck --check-prefix=GLOBAL-SYM %s
// GLOBAL-SYM: NOTYPE GLOBAL DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.global.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=6 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.weak.elf
// RUN: %llvm-readelf -s %t.weak.elf | %FileCheck --check-prefix=WEAK-SYM %s
// WEAK-SYM: NOTYPE WEAK DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.weak.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=7 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.direct.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.direct.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// RUN: %clang -x assembler-with-cpp -DENTRY_KIND=8 -target amdgcn-amd-amdhsa \
// RUN:   -mcpu=gfx1250 -nostdlib %s -o %t.local-notype.elf
// RUN: %llvm-readelf -s %t.local-notype.elf \
// RUN:   | %FileCheck --check-prefix=LOCAL-NOTYPE %s
// LOCAL-NOTYPE: NOTYPE LOCAL DEFAULT {{.*}} protected_entry
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.local-notype.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 | %FileCheck %s

// CHECK: hotswap: error: no safe short-branch gateway for far site
// CHECK: RESULT: ERROR

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

// Without entry protection this zero run is the only reachable gateway and
// its first bytes, including protected_entry, are overwritten.
#if ENTRY_KIND == 0 || ENTRY_KIND == 7
#define PROTECTED_ENTRY .Lprotected_entry
#else
#define PROTECTED_ENTRY protected_entry
#endif

#if ENTRY_KIND == 7
.local direct_source
.type direct_source,@function
direct_source:
  s_branch PROTECTED_ENTRY
.size direct_source, .-direct_source
#endif

#if ENTRY_KIND == 5
.globl protected_entry
#elif ENTRY_KIND == 6
.weak protected_entry
#elif ENTRY_KIND != 0 && ENTRY_KIND != 7
.local protected_entry
#endif
#if ENTRY_KIND == 1
.type protected_entry,@function
#elif ENTRY_KIND == 2
.type protected_entry,@gnu_indirect_function
#elif ENTRY_KIND == 4
.type protected_entry,@object
#endif
PROTECTED_ENTRY:
  .zero 32
#if ENTRY_KIND == 1 || ENTRY_KIND == 2 || ENTRY_KIND == 4
.size protected_entry, .-protected_entry
#endif

// Push the appended trampoline pool beyond s_branch reach.
.rept 40000
  s_mov_b32 s64, s65
.endr

.rodata
.p2align 8
.amdhsa_kernel gateway_kernel
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

#if ENTRY_KIND == 3
.p2align 8
.amdhsa_kernel protected_entry
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
#endif

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
#if ENTRY_KIND == 3
    - .name: protected_entry
      .symbol: protected_entry.kd
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
#endif
.end_amdgpu_metadata
