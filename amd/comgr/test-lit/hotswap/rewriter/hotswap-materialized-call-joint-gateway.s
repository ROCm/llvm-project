// COM: The production corpus contains compiler PC-materialized s30 calls into
// COM: a helper with exact materialized set-PC jumps and an s30 return. Those
// COM: three edge families form one finite component. MC also classifies the
// COM: swap-call as an indirect branch; that generic classification must not
// COM: create an unbounded self-edge after the same call has been proven
// COM: finite. Closing the component keeps external zero padding available as
// COM: a gateway for an otherwise-stranded far eight-byte DS2 patch.

// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: accepted exact materialized-call/canonical-return closure
// LOG-NOT: hotswap: unresolved call target
// LOG-NOT: hotswap: unresolved control-flow target disables
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: ds_load_2addr
// DISASM: ds_load_b64
// DISASM-NEXT: ds_load_b64

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.local joint_helper
.type joint_helper,@function
joint_helper:
  // Match the production helper: preserve its incoming link in callee-saved
  // VGPR lanes, use s30:s31 as scratch, then restore it before returning.
  v_writelane_b32 v40, s30, 0
  v_writelane_b32 v41, s31, 1
  s_mov_b32 s30, 0
  s_mov_b32 s31, 0

  // This exact local set-PC jump is needed to reach the restore/return path.
  s_get_pc_i64 s[4:5]
.Lhelper_pc:
  s_add_nc_u64 s[4:5], s[4:5], .Lhelper_restore-.Lhelper_pc
  s_set_pc_i64 s[4:5]
  s_endpgm
.Lhelper_restore:
  v_readlane_b32 s30, v40, 0
  v_readlane_b32 s31, v41, 1
  s_set_pc_i64 s[30:31]
  s_endpgm
.size joint_helper, .-joint_helper

.globl test_materialized_call_joint_gateway
.p2align 8
.type test_materialized_call_joint_gateway,@function
test_materialized_call_joint_gateway:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2

  // The call target is the exact local helper entry. Its continuation is the
  // following s_endpgm instruction.
  s_get_pc_i64 s[0:1]
.Lcaller_pc:
  s_add_nc_u64 s[0:1], s[0:1], joint_helper-.Lcaller_pc
  s_swap_pc_i64 s[30:31], s[0:1]
  s_endpgm
.size test_materialized_call_joint_gateway, .-test_materialized_call_joint_gateway

// Safe external gateway space. A falsely-unbounded call clears this map and
// makes the far eight-byte patch fail closed.
.fill 64, 1, 0

// Keep the appended trampoline pool beyond one signed s_branch span.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_materialized_call_joint_gateway
  .amdhsa_next_free_vgpr 42
  .amdhsa_next_free_sgpr 42
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_materialized_call_joint_gateway
      .symbol: test_materialized_call_joint_gateway.kd
      .sgpr_count: 42
      .vgpr_count: 42
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
