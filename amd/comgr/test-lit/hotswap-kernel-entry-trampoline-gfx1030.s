// COM: HotSwap redirects kernel descriptors to appended PC-relative entry
// COM: stubs on gfx10.3. Unlike gfx12.5, the gfx1030 stub has no global_wb +
// COM: v_nop marker: it is a bare get-PC / add / addc / set-PC jump back to the
// COM: original kernel entrypoint, spelled with the gfx10-style s_getpc_b64 /
// COM: s_setpc_b64.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 -nostdlib %s -o %t.elf

// COM: Without the flag the rewrite is a no-op copy of the input.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1030 amdgcn-amd-amdhsa--gfx1030 \
// RUN:   --output %t.default.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: cmp %t.elf %t.default.elf
// RUN: %llvm-objdump -d %t.default.elf | %FileCheck --check-prefix=NO-TRAMP %s
// NO-TRAMP-LABEL: <entry_tramp_kernel>:
// NO-TRAMP: s_endpgm
// NO-TRAMP-LABEL: <second_kernel>:
// NO-TRAMP: s_endpgm

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1030 amdgcn-amd-amdhsa--gfx1030 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The gfx1030 stub is a bare PC-relative jump: no global_wb, no v_nop.
// DISASM-NOT: global_wb
// DISASM: s_getpc_b64 s[8:9]
// DISASM-NEXT: s_add_u32 s8
// DISASM-NEXT: s_addc_u32 s9
// DISASM-NEXT: s_setpc_b64 s[8:9]

// COM: Each appended stub gets a <kernel>.stub symbol so a dispatch whose entry
// COM: points at the stub still resolves to a name (e.g. rocgdb info dispatches).
// RUN: %llvm-readelf -s %t.out.elf | %FileCheck --check-prefix=SYMS %s
// SYMS-DAG: FUNC {{.*}} entry_tramp_kernel.stub
// SYMS-DAG: FUNC {{.*}} second_kernel.stub

// COM: The rewrite is idempotent: a second pass detects the existing stubs and
// COM: produces byte-identical output.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1030 amdgcn-amd-amdhsa--gfx1030 \
// RUN:   --entry-trampolines --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
.text
.globl entry_tramp_kernel
.p2align 8
.type entry_tramp_kernel,@function
entry_tramp_kernel:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Lentry_tramp_kernel_end:
.size entry_tramp_kernel, .Lentry_tramp_kernel_end-entry_tramp_kernel

.globl second_kernel
.p2align 8
.type second_kernel,@function
second_kernel:
  s_nop 0
  s_endpgm
.Lsecond_kernel_end:
.size second_kernel, .Lsecond_kernel_end-second_kernel

.rodata
.p2align 8
.amdhsa_kernel entry_tramp_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel second_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: entry_tramp_kernel
      .symbol: entry_tramp_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: second_kernel
      .symbol: second_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
