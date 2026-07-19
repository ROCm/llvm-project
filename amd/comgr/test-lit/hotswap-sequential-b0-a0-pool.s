// COM: A target-state certificate must cover every executable byte in the
// COM: object, including code emitted by an earlier HotSwap pass. A strict B0
// COM: rewrite moves this PC-sensitive tensor instruction into an appended
// COM: executable pool. A later B0-to-A0 pass must not ignore that pool and
// COM: retag the kernel A0.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// COM: Control case: a normal object whose only executable code is `.text`
// COM: remains eligible. Replacing the tensor with an unaligned DS2 also makes
// COM: the successful A0 pass append a target-correct pool. Repeating that
// COM: completed rewrite must use the A0 certificate before the structural
// COM: outside-`.text` guard and return byte-identical output.
// RUN: sed 's/tensor_load_to_lds s\[0:3\], s\[4:11\]/ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2/' %s > %t.control.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.control.s -o %t.control.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.control.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.control.a0.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=CONTROL %s
// CONTROL: growWithTrampolines: appended 1 trampoline
// CONTROL: RESULT: SUCCESS
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.control.a0.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.control.repeat.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=REPEAT %s
// REPEAT: every kernel already reports gfx1250 revision A0
// REPEAT-NOT: executable code outside .text
// REPEAT: RESULT: SUCCESS
// RUN: cmp %t.control.a0.elf %t.control.repeat.elf

// COM: The pool note, not optional per-kernel metadata, carries completion
// COM: state for generated executable code. With metadata removed, a repeated
// COM: B0-to-A0 request must still accept the A0 pool, apply no new patches,
// COM: and return byte-identical output.
// RUN: sed -e 's/tensor_load_to_lds s\[0:3\], s\[4:11\]/ds_load_2addr_b64 v[0:3], v4 offset0:1 offset1:2/' \
// RUN:   -e '/^\.amdgpu_metadata$/,/^\.end_amdgpu_metadata$/d' %s > %t.nometa.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.nometa.s -o %t.nometa.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nometa.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.nometa.a0.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=NOMETA-FIRST %s
// NOMETA-FIRST: target-state 1 provenance
// NOMETA-FIRST: RESULT: SUCCESS
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nometa.a0.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.nometa.repeat.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=NOMETA-REPEAT %s
// NOMETA-REPEAT: applied 0 instruction patches
// NOMETA-REPEAT-NOT: incompatible executable code
// NOMETA-REPEAT: RESULT: SUCCESS
// RUN: cmp %t.nometa.a0.elf %t.nometa.repeat.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.b0.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=B0 %s
// B0: growWithTrampolines: appended 1 trampoline
// B0: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.b0.elf | %FileCheck --check-prefix=B0-DIS %s
// B0-DIS-LABEL: <sequential_tensor>:
// B0-DIS: s_branch
// B0-DIS: Disassembly of section :
// B0-DIS: tensor_load_to_lds

// COM: Until HotSwap can retarget every executable section, the generalized
// COM: safe behavior is to reject this composition rather than issue a false
// COM: object-wide A0 certificate.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.b0.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=A0 %s
// A0: was produced for an incompatible target stepping
// A0: RESULT: ERROR

// COM: Pool provenance is validated before the per-kernel A0 metadata fast
// COM: path. A stale or forged A0 metadata tag must not certify a B0 pool.
// RUN: sed 's/\.gfx1250_revision: B0/.gfx1250_revision: A0/' %s > %t.a0meta.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.a0meta.s -o %t.a0meta.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.a0meta.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.a0meta.b0pool.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=STALE-FIRST %s
// STALE-FIRST: target-state 2 provenance
// STALE-FIRST: RESULT: SUCCESS
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.a0meta.b0pool.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=STALE-A0 %s
// STALE-A0-NOT: every kernel already reports gfx1250 revision A0
// STALE-A0: was produced for an incompatible target stepping
// STALE-A0: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl sequential_tensor
.p2align 8
.type sequential_tensor,@function
sequential_tensor:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
.size sequential_tensor, .-sequential_tensor

.rodata
.p2align 8
.amdhsa_kernel sequential_tensor
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: sequential_tensor
      .symbol: sequential_tensor.kd
      .gfx1250_revision: B0
      .sgpr_count: 16
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
