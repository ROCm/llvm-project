; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Lift fixture for FLAT `global_load_async_to_lds_b{8,32,64,128}` on the
; CROSS-TARGET arm (gfx1250 -> gfx942). Pins the synchronous per-lane
; emulation in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` when
; `ctx.targetIsa.hasTensorOps` is false. Companion fixture to
; `global_load_async_to_lds_same_target.ll`, which pins the
; same-target (gfx1250 -> gfx1250) intrinsic-emit shape.
;
; === Why this is an emulation, not the prior loud refusal ===
;
; Pre-2026-04-24 (see the git-log entry accompanying this fixture
; change), the cross-target arm refused the lift loudly with a
; `RaiseFailure::unsupportedInstructionForm` diagnostic that cited the
; gfx1250-only asynccnt unit and the gated `amdgcn.global.load.async.
; to.lds.b*` intrinsic. That posture was correct in the sense of
; avoiding silent miscompile, but it blocked every `matmul_ogs_*`
; MoE expert-GEMM kernel in the GPT-OSS surface (4 kernels,
; runtime-dominant in inference) end-to-end on gfx942. Every one
; of those kernels compiled without errors up to the first
; async-DMA instruction, then refused.
;
; The DOCUMENTED semantic trade-off (spelled out verbatim on the
; SemOp doc block in `semop.hpp`): the synchronous `load` + `store`
; chain produces bit-identical per-lane LDS state to what the source
; async DMA produces AFTER the companion `s_wait_asynccnt 0`. The
; ONLY information lost is the **pipelining overlap** between the
; async DMA and unrelated VMEM / LDS ops in the wave's own
; instruction stream; on gfx942 the emulation serialises those.
; This is a throughput regression, not a correctness regression.
; Kernels that depend on observable effects under a partially-
; elapsed asynccnt (e.g. a hand-written pipeliner polling asynccnt
; state) are NOT in the GPT-OSS corpus and remain out of scope;
; they would have to be hand-written to produce instructions that
; LLVM IR cannot express anyway.
;
; === ISA pragma reference (verbatim) ===
;
; From `hotswap/docs/manuals/instruction_manual.pdf §13.6.{9,10,11,12}`
; (MI400 Shader Instruction Set, sp3), identical across all four
; widths except the LDS-store byte count:
;
;   pragma "vector" do
;     dsaddr  = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
;     memaddr = ADDR;   // CalcGlobalAddr(VADDR, SADDR, IOFFSET)
;     LDS[dsaddr].bN = MEM[memaddr].bN   (N = 8 / 32 / 64 / 128)
;   endpragma
;
; Two key observations the emulation relies on:
;
;   1. `pragma "vector" do` runs the body per-active-lane; the
;      raiser wraps the `load` + `store` pair in `emitUnderExec` so
;      inactive lanes skip the whole memory round-trip.
;
;   2. `INST_OFFSET` applies to BOTH the LDS address and the global
;      address.  For this HIP fixture all offsets are zero, so the
;      GEP chain is elided.  The handler has a non-zero-offset
;      branch that GEPs the offset onto both pointers via
;      `CreateGEP(i8Ty, ptr, i64 offset)`; a future fixture
;      exercising non-zero flat_offset would pin that branch.
;
; === Per-width emission ===
;
; The HIP fixture compiles the four clang builtins
; `__builtin_amdgcn_global_load_async_to_lds_b{8,32,64,128}` into
; SADDR-form instructions with `scale_offset` enabled (the
; assembler emits `scale_offset` because the per-lane VGPR offset
; is the thread-id times the access element size).  The emulation
; materialises `scale_offset` as `mul i64 %voff_zext, N` for
; N = 1 / 4 / 8 / 16 respectively (see
; `decode.cpp::decodeScaleOffset` for how the cpol bit becomes
; `DecodedInst::hasScaleOffset`).  For b8, accessBytes = 1 makes
; scaled and unscaled offsets identical; the assembler / decoder
; emits the plain `saddr + voff` shape, so the b8 section pins the
; byte-width load/store rather than a no-op multiply.
;
; The LDS-base VGPR holds the per-lane i32 address (lds_base +
; tid*elemBytes on this fixture); the emulation casts it to
; `ptr addrspace(3)` via `inttoptr i32`, matching the same-target
; arm's LDS-pointer shape (see
; `global_load_async_to_lds_same_target.ll`) so IR-shape reviewers
; can visually diff the two arms at the LDS-base cast.
;
; Access type per width — larger widths are lifted as vectors of
; i32 rather than a single `iN`, mirroring
; `GLOBAL_LOAD_DWORDX{2,3,4}`'s handling of the same aggregate
; shape; the backend picks `global_load_{dword,dwordx2,dwordx4}` /
; `ds_store_{b32,b64,b128}` on gfx942 from the aligned-load
; attribute below:
;
;   b32  : i32
;   b64  : <2 x i32>
;   b128 : <4 x i32>
;   b8   : i8
;
; The natural alignment (accessBytes = 4 / 8 / 16) is attached to
; both the load and store so the backend's memop-alignment-derived
; codegen picks the right dword / dwordx2 / dwordx4 opcode.

; LDS-destination drop gate.  gfx12 `global_load_async_to_lds` drops any lane
; whose LDS-destination offset is outside the LDS aperture; Triton uses an
; INT_MAX (0x7fffffff) LDS-dest sentinel to predicate masked GEMM-tile rows
; (whose global tile address is intentionally OOB and whose LDS slot is
; pre-initialised to the `other` value).  The synchronous emulation MUST
; replicate the drop -- otherwise the masked lane's OOB global address is
; dereferenced and faults whenever it is unmapped.  Each width below therefore
; gates the emulated load+store on the LDS destination being within the 64 KiB
; LDS capacity (`icmp ult i32 %lds_off, 65536` + a `br` into the do-block).
;
; ----- b32 ----- (first load in the HIP kernel)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_lds_inb{{[0-9]*}} = icmp ult i32 {{.*}}, 65536
; IR: br i1 %async_lds_inb{{[0-9]*}}
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %{{[0-9]+}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 4

; ----- b64 ----- (second load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 8
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <2 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 8
; IR: store <2 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 8

; ----- b128 ----- (third load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 16
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <4 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 16
; IR: store <4 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 16

; ----- b8 ----- (fourth load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %voff_zext{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load i8, ptr addrspace(1) %{{[0-9]+}}, align 1
; IR: store i8 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 1

; ----- Negative assertions -----
;
; The emulation MUST NOT emit the native intrinsic on the cross-
; target arm (the intrinsic is gfx1250-only and would fail isel on
; gfx942).  A regression that accidentally took the same-target
; path on a non-gfx1250 target would surface here as an IR shape
; containing the intrinsic name.

; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b8
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b64
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b128

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx1250 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SAME
;
; Lift fixture for FLAT `global_load_async_to_lds_b{8,32,64,128}` —
; the same-target (gfx1250 → gfx1250) intrinsic-emit path. Pins
; the principled lift in transpiler/handle_flat.cpp under
; `SemOp::GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `global_load_async_to_lds.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; The FLATInstructions.td `FLAT_Global_Load_LDS_Pseudo<IsAsync=1>`
; multiclass yields three operand-shape variants — plain
; (vdst:VGPR_32, vaddr:VGPR_64, off:imm, cpol:imm) and SADDR
; (vdst:VGPR_32, saddr:SGPR_64, vaddr:VGPR_32, off:imm, cpol:imm)
; for each width. The fixture's HIP source uses the clang
; builtins `__builtin_amdgcn_global_load_async_to_lds_b{8,32,64,128}`
; which compile to the plain VFLAT 0x60-0x62 reals (vdst=v1,
; vaddr=v0, saddr=s[0:1]/s[6:7]/s[4:5] — the kernel-arg pointers
; landed in the SGPR base + per-lane VGPR offset shape, which the
; AMDGPU disassembler prints as the SADDR variant).
;
; The matching LLVM intrinsic family
; (IntrinsicsAMDGPU.td:3939-3946, all sharing `AMDGPUAsyncGlobalLoadToLDS`
; on line 3904) is
;
;   void llvm.amdgcn.global.load.async.to.lds.b{8,32,64,128}(
;       ptr addrspace(1) %gaddr,
;       ptr addrspace(3) %laddr,
;       i32 immarg      %offset,
;       i32 immarg      %cpol)
;
; The handler:
;   * casts `vdst` (a per-lane VGPR_32 carrying the LDS i32 base)
;     via `inttoptr i32 %vgpr to ptr addrspace(3)` (named
;     `lds_ptr*` in the emitted IR);
;   * decodes the global address via the shared FLAT `decodeFlatAddr`
;     helper (plain → vaddr-only, SADDR → saddr+vaddr) into a
;     `ptr addrspace(1)`;
;   * threads the FLAT `offset` immediate and the `cpol` immediate
;     through as the trailing `i32 immarg` pair (cpol = 0x800 here
;     because the assembler emits `scale_offset` for the
;     scaled-saddr path);
;   * wraps the call in `ctx.emitUnderExec` so wave-divergent EXEC
;     is honoured before the side-effecting DMA.
;
; We pin the per-width call shape and the LDS-pointer cast it
; consumes. Drift indicators:
;   * If a future LLVM rename swaps the intrinsic name (e.g. drops
;     the `async.` infix) the IR check fails immediately and
;     pinpoints the rename rather than letting a silently mis-named
;     intrinsic reach the backend.
;   * If the LDS-base operand is lowered as a `<n x i32>` vector
;     (or any non-`ptr addrspace(3)` shape), the matching
;     `inttoptr i32 ... to ptr addrspace(3)` line goes missing and
;     FileCheck reports the exact divergence.

; b32: per-lane LDS i32 base via inttoptr i32 → ptr addrspace(3),
; then the b32 async DMA call.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b32(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b64: same shape, b64 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b64(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b128: same shape, b128 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b128(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b8: same shape, b8 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b8(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_async_to_lds_kernel
	.p2align	8
	.type	global_load_async_to_lds_kernel,@function
global_load_async_to_lds_kernel:        ; @global_load_async_to_lds_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v1, v0, 2, 0x600
	s_wait_kmcnt 0x0
	global_load_async_to_lds_b32 v1, v0, s[4:5] scale_offset
	v_lshl_add_u32 v1, v0, 3, 0x400
	global_load_async_to_lds_b64 v1, v0, s[6:7] scale_offset
	v_lshlrev_b32_e32 v1, 4, v0
	global_load_async_to_lds_b128 v1, v0, s[8:9] scale_offset
	v_add_nc_u32_e32 v1, 0x700, v0
	global_load_async_to_lds_b8 v1, v0, s[10:11]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_async_to_lds_kernel
		.amdhsa_group_segment_fixed_size 1856
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_async_to_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         global_load_async_to_lds_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
