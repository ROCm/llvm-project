//===- raiser.cpp - Transpiler MC -> LLVM IR raiser ----------------------===//
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

#include "transpiler/raiser/raiser.h"

#include "transpiler/decoder/amdgpu-formats.h"
#include "transpiler/decoder/decode.h"
#include "transpiler/decoder/mc-state.h"
#include "transpiler/decoder/opcode-map.h"
#include "transpiler/decoder/setpc-analysis.h"
#include "transpiler/raiser/handlers.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"
#include "transpiler/raiser/wave-projection.h"

#include "comgr.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
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
#include "llvm/MC/MCSubtargetInfo.h"
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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace COMGR::transpiler {

// Address space the kernarg segment lives in.
constexpr unsigned ConstantAddressSpace = 4;

// Identifier the raised module carries. A code object names no module of its
// own, and one raise holds every kernel of it.
constexpr StringLiteral kRaisedModuleName = "transpiler.raised";

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
                                       const MCSubtargetInfo &SourceSTI) {
  const unsigned DefaultDenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  const unsigned Float32DenormalMode = AMDHSA_BITS_GET(
      Meta.ComputePgmRsrc1, amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  const DenormalFPEnv FPEnv(denormalMode(DefaultDenormalMode),
                            denormalMode(Float32DenormalMode));
  F.addFnAttr(Attribute::get(F.getContext(), Attribute::DenormalFPEnv,
                             FPEnv.toIntValue()));

  if (!SourceSTI.hasFeature(AMDGPU::FeatureDX10ClampAndIEEEMode)) {
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
                               const MCSubtargetInfo &SourceSTI) {
  LLVMContext &C = M.getContext();
  SmallVector<Type *> ParamTys;
  if (Meta.KernargSegmentSize > 0)
    ParamTys.push_back(PointerType::get(C, ConstantAddressSpace));

  FunctionType *FuncTy =
      FunctionType::get(Type::getVoidTy(C), ParamTys, /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);
  setFloatingPointAttributes(*F, Meta, SourceSTI);

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
  OperandResolver Op{Ctx, Di};

  if (Di.VOPD)
    return handleVOPD(Ctx, Di);

  if (Di.TargetSpecificFlags & SOP1)
    return handleSOP1(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOP2)
    return handleSOP2(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPC)
    return handleSOPC(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPK)
    return handleSOPK(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SOPP)
    return handleSOPP(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & SMRD)
    return handleSMEM(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & FLAT)
    return handleFLAT(Ctx, Di, Op);
  if (Di.TargetSpecificFlags & DS)
    return handleDS(Ctx, Di);

  constexpr uint64_t VOP1EncodingMask = VOP1 | VOP3 | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP1EncodingMask) == VOP1)
    return handleVOP1(Ctx, Di, Op);

  constexpr uint64_t VOP2EncodingMask =
      VOP2 | VOP3 | VOP3P | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP2EncodingMask) == VOP2) {
    return handleVOP2(Ctx, Di, Op);
  }

  constexpr uint64_t VOP3EncodingMask =
      VOP3 | VOP3P | VOPC | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOP3EncodingMask) == VOP3)
    return handleVOP3(Ctx, Di, Op);

  constexpr uint64_t VOPCEncodingMask =
      VOPC | VOP3 | VOP3P | DPP | SDWA | VOPD3;
  if ((Di.TargetSpecificFlags & VOPCEncodingMask) == VOPC) {
    return handleVOPC(Ctx, Di, Op);
  }

  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedInstructionForm,
      strippedMnemonic(Ctx.MC, Di.Inst), Di.Offset,
      formatName(Di.TargetSpecificFlags));
}

// The MC layer for one ISA. `Role` names which end of the raise this is, and
// only reaches diagnostics.
namespace {
struct IsaContext {
  MCState MC;
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

  // The target side reads only the subtarget and registered target behind the
  // machine, and pays for a disassembler and a printer it never uses. That is
  // one extra MC stack per raise, against a second way of standing a subtarget
  // up that has to be kept in step with this one.
  Expected<MCState> MC = initMCState(Cpu);
  if (!MC)
    return MC.takeError();

  return IsaContext{std::move(*MC), Cpu.str()};
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

// A half-open range of text offsets the decode has read.
struct Extent {
  uint64_t Begin = 0;
  uint64_t End = 0;

  bool contains(uint64_t Offset) const {
    return Offset >= Begin && Offset < End;
  }
};

// Fold a second decode into `Base`, keeping the instructions in source order.
// The two are decoded from disjoint extents, so neither carries an instruction
// the other already has.
static void mergeDecoded(DecodeResult &Base, DecodeResult &&Extra) {
  llvm::move(Extra.Insts, std::back_inserter(Base.Insts));
  sort(Base.Insts, [](const DecodedInst &A, const DecodedInst &B) {
    return A.Offset < B.Offset;
  });
  set_union(Base.BlockStarts, Extra.BlockStarts);
}

// Raise one kernel into `M`. Everything this allocates -- the projection, the
// builder, the register file behind the context -- describes that one kernel
// and dies with the call; only the emitted function outlives it.
static Error raiseKernel(const RaiseEnvironment &Env, Module &M,
                         const TextSection &Text, const KernelRequest &Kernel,
                         ArrayRef<KernelSymbolExtent> FunctionExtents) {
  const KernelMeta &Meta = Kernel.Meta;
  Expected<DecodeResult> Decoded = decodeKernel(
      Env.Source.MC, Env.OpcMap, Text.Bytes, Kernel.StartOffset,
      Kernel.EndOffset == 0 ? std::nullopt : std::optional(Kernel.EndOffset));
  if (!Decoded)
    return Decoded.takeError();

  // A jump through a register names no offset the decode could follow, so the
  // blocks such a jump leads to are only known once the values behind them
  // have been worked out. A call may also leave the kernel entirely, for an
  // outlined helper the decode never saw; following it means decoding that
  // helper and asking again, until nothing new turns up.
  SmallVector<Extent> Covered{{Kernel.StartOffset, Kernel.EndOffset == 0
                                                       ? Text.Bytes.size()
                                                       : Kernel.EndOffset}};
  std::optional<SetPcAnalysis> SetPc;
  for (;;) {
    Expected<SetPcAnalysis> Analyzed =
        analyzeSetPc(Decoded->Insts, Decoded->BlockStarts, Kernel.StartOffset,
                     Env.Source.MC);
    if (!Analyzed)
      return Analyzed.takeError();
    SetPc = std::move(*Analyzed);

    // A target the decode is not widened to cover keeps the refusal the
    // analysis already recorded for the transfer reaching it, which the raise
    // reports when it gets there. Neither kind below is worth widening for: a
    // target inside an extent already decoded points into the middle of an
    // instruction, and reading those bytes again decodes them the same way; a
    // target no function symbol covers says neither where to start reading nor
    // how much.
    bool Followed = false;
    for (uint64_t Target : SetPc->UndecodedTargets) {
      if (any_of(Covered, [&](const Extent &E) { return E.contains(Target); }))
        continue;
      const KernelSymbolExtent *Callee =
          find_if(FunctionExtents, [&](const KernelSymbolExtent &E) {
            return Target >= E.Offset && Target < E.Offset + E.Size;
          });
      if (Callee == FunctionExtents.end())
        continue;

      // The callee is decoded whole and from its own entry rather than from
      // the offset that reached it, so its block starts are the ones its own
      // code implies.
      Expected<DecodeResult> CalleeDecoded =
          decodeKernel(Env.Source.MC, Env.OpcMap, Text.Bytes, Callee->Offset,
                       Callee->Offset + Callee->Size);
      if (!CalleeDecoded)
        return CalleeDecoded.takeError();
      Covered.push_back({Callee->Offset, Callee->Offset + Callee->Size});
      mergeDecoded(*Decoded, std::move(*CalleeDecoded));
      Followed = true;
    }
    if (!Followed)
      break;
  }

  // Merging the block starts here, before any block is made, is what lets the
  // handler find the block its jump targets.
  Decoded->BlockStarts.insert(SetPc->ExtraBlockStarts.begin(),
                              SetPc->ExtraBlockStarts.end());

  LLVMContext &C = M.getContext();

  // Replication is the only projection policy the raiser can select: a target
  // lane reads the source EXEC bit of the source lane it stands in for. What
  // that costs when the two wave sizes differ is the policy's own business.
  ReplicationProjection Projection(*Env.Source.MC.SubtargetInfo,
                                   *Env.Target.MC.SubtargetInfo,
                                   Type::getInt32Ty(C), Type::getInt64Ty(C));
  Projection.setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);

  Function *F =
      declareKernel(M, Kernel.Name, Meta, *Env.Source.MC.SubtargetInfo);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);

  Expected<RaiseContext> Ctx = RaiseContext::create(
      B, Projection, Env.Source.MC, *SetPc, Meta, Text.Bytes, Text.Address,
      Text.ImageSections, Kernel.StartOffset, Kernel.EndOffset);
  if (!Ctx)
    return Ctx.takeError();

  // A block per recovered block start, all of them made before any instruction
  // is raised so a branch reaching forward finds the block it targets. The
  // kernel entry gets one too, rather than raising into the entry block the
  // allocas live in: a branch back to the first instruction would otherwise
  // give the entry block a predecessor, which LLVM does not allow.
  for (uint64_t Start : Decoded->BlockStarts)
    Ctx->defineBB(Start, BasicBlock::Create(C, formatv("bb_{0:x}", Start), F));

  // A followed callee can sit anywhere in the text section, the kernel's own
  // entry included, so where the raise starts is named rather than left to be
  // whichever block the first raised instruction leads.
  B.CreateBr(Ctx->lookupBB(Kernel.StartOffset));

  for (const DecodedInst &Di : Decoded->Insts) {
    BasicBlock *Open = B.GetInsertBlock();
    if (Decoded->BlockStarts.count(Di.Offset)) {
      BasicBlock *Next = Ctx->lookupBB(Di.Offset);
      // A source block ending in something other than a control transfer
      // reaches the block that follows it, which LLVM states as a branch.
      if (!Open->hasTerminator())
        B.CreateBr(Next);
      B.SetInsertPoint(Next);
    } else if (Open->hasTerminator()) {
      // An instruction trailing a control transfer without leading a block
      // start of its own is reached by nothing, and needs a block anyway for
      // its handler to raise into.
      B.SetInsertPoint(
          BasicBlock::Create(C, formatv("unreached_{0:x}", Di.Offset), F));
    }

    Ctx->registers().computeVGPRAdjust(Di);
    if (Error Err = raiseInst(*Ctx, Di))
      return Err;
  }

  // Execution reaching the end of the extent means the code is truncated or
  // the extent is misbounded. Closing the block with a return instead would
  // hand back a kernel that reads as having run to completion. Every earlier
  // block is terminated on the way out of it, so the open one is the only
  // block that can still be missing a terminator here.
  if (!B.GetInsertBlock()->hasTerminator())
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
                                ArrayRef<KernelRequest> Kernels,
                                ArrayRef<KernelSymbolExtent> FunctionExtents) {
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
    if (Error Err = raiseKernel(*Env, M, Text, Kernel, FunctionExtents))
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

} // namespace COMGR::transpiler
