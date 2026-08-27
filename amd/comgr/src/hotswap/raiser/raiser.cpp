//===- raiser.cpp - Hotswap MC -> LLVM IR raiser -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decodes each requested kernel's .text into a typed `DecodedInst` stream,
// dispatches each decoded instruction to its per-format handler, promotes the
// register-file allocas to SSA, and verifies the module of `amdgpu_kernel`
// functions this produces. The MC layer the decode runs on is built once and
// shared, since the kernels come from one code object.
//
// A raise reads two ISAs. The source one is what the code object was compiled
// for, and the decode is written in its terms. The target one is what the
// raised IR will be lowered for, and the wave projection reads it to translate
// a source lane into the target lane that runs it.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser/raiser.h"

#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/opcode-map.h"
#include "hotswap/raiser/handlers.h"
#include "hotswap/raiser/op-resolver.h"
#include "hotswap/raiser/raise-context.h"
#include "hotswap/raiser/raise_failure.h"
#include "hotswap/raiser/wave-projection.h"

#include "comgr.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace COMGR::hotswap {

// Address space the kernarg segment lives in.
constexpr unsigned ConstantAddressSpace = 4;

// Identifier the raised module carries. A code object names no module of its
// own, and one raise holds every kernel of it.
constexpr StringLiteral kRaisedModuleName = "hotswap.raised";

// Minimum kernarg segment alignment the AMDGPU ABI mandates.
constexpr Align KernargSegmentAlign = Align::Constant<16>();

// The bare AMDGPU processor name `Isa` denotes. Callers pass either that name
// (`gfx942`) or a canonical target identifier
// (`amdgcn-amd-amdhsa--gfx942:xnack-`); the MC layer accepts only the former.
// The result points into `Isa`.
static StringRef processorName(StringRef Isa) {
  TargetIdentifier Ident;
  if (parseTargetIdentifier(Isa, Ident) == AMD_COMGR_STATUS_SUCCESS)
    return Ident.Processor;
  return Isa;
}

/// Return the LLVM denormal mode represented by an AMDHSA descriptor field.
static DenormalMode denormalMode(unsigned HardwareMode) {
  using Kind = DenormalMode::DenormalModeKind;
  switch (HardwareMode) {
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC_DST:
    return {Kind::PreserveSign, Kind::PreserveSign};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_DST:
    return {Kind::PreserveSign, Kind::IEEE};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_SRC:
    return {Kind::IEEE, Kind::PreserveSign};
  case amdhsa::FLOAT_DENORM_MODE_FLUSH_NONE:
    return {Kind::IEEE, Kind::IEEE};
  }
  llvm_unreachable("invalid hardware denormal mode");
}

/// Attach the floating-point attributes represented by the source descriptor.
static void setFloatingPointAttributes(Function &F, const KernelMeta &Meta,
                                       const ISAProfile &SourceProfile) {
  const unsigned DefaultDenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  const unsigned Float32DenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  const DenormalFPEnv FPEnv(denormalMode(DefaultDenormalMode),
                            denormalMode(Float32DenormalMode));
  F.addFnAttr(Attribute::get(F.getContext(), Attribute::DenormalFPEnv,
                             FPEnv.toIntValue()));

  if (!SourceProfile.hasDx10ClampAndIeeeMode()) {
    return;
  }

  const bool Dx10Clamp =
      AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                      amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);
  const bool IeeeMode =
      AMDHSA_BITS_GET(Meta.ComputePgmRsrc1,
                      amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);
  F.addFnAttr("amdgpu-dx10-clamp", Dx10Clamp ? "true" : "false");
  F.addFnAttr("amdgpu-ieee", IeeeMode ? "true" : "false");
}

// Declare the lifted kernel: one opaque parameter spanning the source kernarg
// segment, so the emitted descriptor reports the source segment size and the
// ABI alignment. The raised body reads arguments as ordinary loads off the
// kernarg pointer, at the byte offsets the source metadata gives them.
static Function *declareKernel(Module &M, StringRef KernelName,
                               const KernelMeta &Meta,
                               const ISAProfile &SourceProfile) {
  LLVMContext &C = M.getContext();
  SmallVector<Type *> ParamTys;
  if (Meta.KernargSegmentSize > 0)
    ParamTys.push_back(PointerType::get(C, ConstantAddressSpace));

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), ParamTys, /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  setFloatingPointAttributes(*F, Meta, SourceProfile);

  if (Meta.KernargSegmentSize > 0) {
    // AMDGPULowerKernelArguments honors the `align` parameter attribute only on
    // a byref kernel argument; without `byref` the segment would take the array
    // type's natural one-byte alignment.
    Type *SegmentTy =
        ArrayType::get(Type::getInt8Ty(C), Meta.KernargSegmentSize);
    F->addParamAttr(0, Attribute::getWithByRefType(C, SegmentTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, KernargSegmentAlign));
    F->getArg(0)->setName("kernarg_segment");
  }

  // The host fills the kernarg buffer from the source metadata and leaves no
  // room past the source segment, so the target ABI's hidden-argument block
  // must not be appended to it.
  F->addFnAttr("amdgpu-no-implicitarg-ptr");

  // Both attributes below take a "min,max" range, and both source sizes are
  // exact, so each is written as a range of one.
  //
  // Pin the block to the size the source kernel declared, so the backend lays
  // out workitem ids the way the source binary did.
  F->addFnAttr("amdgpu-flat-work-group-size",
               formatv("{0},{0}", Meta.MaxFlatWorkgroupSize).str());
  if (Meta.GroupSegmentFixedSize > 0) {
    // The raiser addresses LDS by absolute offset rather than through a
    // GlobalVariable, so without this the backend would emit
    // group_segment_fixed_size = 0 and treat every LDS access as out of
    // segment.
    F->addFnAttr("amdgpu-lds-size",
                 formatv("{0},{0}", Meta.GroupSegmentFixedSize).str());
  }
  return F;
}

// Lower one decoded instruction into `Ctx`'s current insertion point, routing
// it by instruction format. A format with no handler is refused rather than
// lowered as something else.
static Error raiseInst(RaiseContext &Ctx, const DecodedInst &Di) {
  using namespace AmdgpuFormat;
  OpResolver Op{Ctx, Di};

  if (Di.TargetSpecificFlags & SOP1)
    return handleSOP1(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOP2)
    return handleSOP2(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPC)
    return handleSOPC(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPP)
    return handleSOPP(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SMRD)
    return handleSMEM(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & FLAT)
    return handleFLAT(Ctx, Di, Op);

  constexpr uint64_t VOP2EncodingMask =
      VOP2 | VOP3 | VOP3P | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP2EncodingMask) == VOP2) {
    return handleVOP2(Ctx, Di, Op);
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

// The MC layer for one ISA and the profile read off it. The profile points
// into the MC state, so the two are kept together and move together. `Role`
// names which end of the raise this is, and only reaches diagnostics.
namespace {
struct IsaContext {
  MCState MC;
  ISAProfile Profile;
  // Bare AMDGPU processor the MC layer was built for.
  std::string Cpu;

  static Expected<IsaContext> create(StringRef Isa, StringRef Role);
};
} // namespace

Expected<IsaContext> IsaContext::create(StringRef Isa, StringRef Role) {
  // Reject a bad ISA before reaching the MC stack: createMCSubtargetInfo
  // accepts an unknown name and returns a featureless subtarget, and the
  // failure only surfaces inside createMCDisassembler, which aborts the
  // process instead of returning.
  StringRef Cpu = processorName(Isa);
  if (AMDGPU::parseArchAMDGCN(Cpu) == AMDGPU::GK_NONE)
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 Role + " ISA '" + Isa +
                                     "' does not name an AMDGPU GPU");

  // The target side reads only the subtarget behind the profile and the
  // registered target behind the machine, and pays for a disassembler and a
  // printer it never uses. That is one extra MC stack per raise, against a
  // second way of standing a subtarget up that has to be kept in step with
  // this one.
  Expected<MCState> MC = initMCState(Cpu);
  if (!MC)
    return MC.takeError();

  ISAProfile Profile = ISAProfile::fromSubtarget(*MC->SubtargetInfo);
  return IsaContext{std::move(*MC), Profile, Cpu.str()};
}

// What every kernel of one raise runs against: the ISA the code object was
// compiled for, the ISA it is being raised onto, and the opcode map built over
// the source MC layer. Built once per raise and outlives each kernel's context.
namespace {
struct RaiseEnvironment {
  IsaContext Source;
  IsaContext Target;
  OpcodeMap OpcMap;

  static Expected<RaiseEnvironment> create(StringRef SourceIsa,
                                           StringRef TargetIsa);
};
} // namespace

Expected<RaiseEnvironment> RaiseEnvironment::create(StringRef SourceIsa,
                                                    StringRef TargetIsa) {
  Expected<IsaContext> Source = IsaContext::create(SourceIsa, "source");
  if (!Source)
    return Source.takeError();

  Expected<IsaContext> Target = IsaContext::create(TargetIsa, "target");
  if (!Target)
    return Target.takeError();

  RaiseEnvironment Env{std::move(*Source), std::move(*Target), OpcodeMap()};
  Env.OpcMap.build(*Env.Source.MC.InstrInfo);
  return Env;
}

// Raise one kernel into `M`. Everything this allocates -- the projection, the
// builder, the register file behind the context -- describes that one kernel
// and dies with the call; only the emitted function outlives it.
static Error raiseKernel(const RaiseEnvironment &Env, Module &M,
                         const TextSection &Text, const KernelRequest &Kernel) {
  const KernelMeta &Meta = Kernel.Meta;
  Expected<DecodeResult> Decoded = decodeKernel(
      Env.Source.MC, Env.OpcMap, Text.Bytes, Kernel.StartOffset,
      Kernel.EndOffset == 0 ? std::nullopt : std::optional(Kernel.EndOffset));
  if (!Decoded)
    return Decoded.takeError();

  LLVMContext &C = M.getContext();

  // Replication is the only projection policy the raiser can select: a target
  // lane reads the source EXEC bit of the source lane it stands in for. What
  // that costs when the two wave sizes differ is the policy's own business.
  ReplicationProjection Projection(Env.Source.Profile, Env.Target.Profile,
                                   Type::getInt32Ty(C), Type::getInt64Ty(C));
  Projection.setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);

  Function *F = declareKernel(M, Kernel.Name, Meta, Env.Source.Profile);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);

  // The kernel entry is the only block start the decoder reports, so the whole
  // kernel raises into one block. A second block start means a handler routed
  // below would have to branch, and none of them can.
  assert(Decoded->BlockStarts.size() <= 1 &&
         "no dispatched instruction format recovers a block start");

  Expected<RaiseContext> Ctx = RaiseContext::create(
      B, Projection, Env.Source.MC, Meta, Text.Bytes, Text.Address,
      Text.ImageSections, Kernel.StartOffset, Kernel.EndOffset);
  if (!Ctx)
    return Ctx.takeError();

  for (const DecodedInst &Di : Decoded->Insts) {
    Ctx->registers().computeVGPRAdjust(Di);
    if (Error Err = raiseInst(*Ctx, Di))
      return Err;
  }

  // Execution reaching the end of the extent means the code is truncated or
  // the extent is misbounded. Closing the block with a return instead would
  // hand back a kernel that reads as having run to completion.
  if (!Entry->hasTerminator())
    return RaiseFailure::general(
        RaiseFailureReason::UnterminatedKernelExtent,
        "kernel extent ends without an instruction that ends the program");

  DominatorTree DT(*F);
  AssumptionCache AC(*F);
  SmallVector<AllocaInst *> Allocas;
  Ctx->registers().collectAllocas(Allocas);
  PromoteMemToReg(Allocas, DT, &AC);
  return Error::success();
}

Expected<RaiseResult> raiseToIR(const TextSection &Text, StringRef SourceIsa,
                                StringRef TargetIsa,
                                ArrayRef<KernelRequest> Kernels) {
  Expected<RaiseEnvironment> Env =
      RaiseEnvironment::create(SourceIsa, TargetIsa);
  if (!Env)
    return Env.takeError();

  RaiseResult Result;
  Result.Ctx = std::make_unique<LLVMContext>();
  Result.Module = std::make_unique<Module>(kRaisedModuleName, *Result.Ctx);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple(kAMDGPUTriple));

  // A module with no data layout leaves every consumer to assume one, so take
  // the AMDGPU layout from a machine built for the processor the raised IR
  // will be lowered for. That machine is also what names the target here: the
  // triple carries no processor, and the raiser emits no target instructions
  // of its own for one to appear in.
  TargetOptions Opts;
  std::unique_ptr<TargetMachine> TM(Env->Target.MC.Target->createTargetMachine(
      Triple(kAMDGPUTriple), Env->Target.Cpu, /*Features=*/"", Opts,
      Reloc::PIC_));
  if (!TM)
    return RaiseFailure::general(
        RaiseFailureReason::TargetMachineCreationFailed,
        "no target machine for '" + Env->Target.Cpu + "'");
  M.setDataLayout(TM->createDataLayout());

  // A refusal is raised where the offending instruction is, which is below the
  // point that knows which kernel of the batch is being raised, so the name and
  // the ISA pair are attached here.
  for (const KernelRequest &Kernel : Kernels)
    if (Error Err = raiseKernel(*Env, M, Text, Kernel))
      return RaiseFailure::withOrigin(std::move(Err), Kernel.Name,
                                      Env->Source.Cpu, Env->Target.Cpu);

  // Verify once the module is whole: a kernel is only well-formed together
  // with the intrinsic declarations its neighbours may also have added.
  std::string VerifyErr;
  raw_string_ostream VerifyOs(VerifyErr);
  if (verifyModule(M, &VerifyOs))
    return RaiseFailure::general(RaiseFailureReason::IRVerificationFailed,
                                 VerifyErr);

  return Result;
}

} // namespace COMGR::hotswap
