//===- handle-valu-output-mods.cpp - VOP3 output modifier guards ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-output-mods.h"

#include "canonical-op.h"

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

bool readNamedImm(const DecodedInst &Di, AMDGPU::OpName Name, int64_t &Out) {
  int Idx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  if (Idx < 0 || static_cast<unsigned>(Idx) >= Di.Inst.getNumOperands())
    return false;
  const MCOperand &Op = Di.Inst.getOperand(static_cast<unsigned>(Idx));
  if (!Op.isImm())
    return false;
  Out = Op.getImm();
  return true;
}

std::optional<int64_t> readNamedImmOperand(const DecodedInst &Di,
                                           AMDGPU::OpName Name) {
  int64_t Value = 0;
  if (!readNamedImm(Di, Name, Value))
    return std::nullopt;
  return Value;
}

} // namespace

bool requireDefaultVOP3OutputMods(const DecodedInst &Di, HandlerResult &Hr,
                                  StringRef DiagnosticName,
                                  VOP3OutputModPresence Presence,
                                  VOP3OutputModDiag Diag) {
  const int ClampIdx =
      AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
  const int OmodIdx =
      AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::omod);

  if (Presence == VOP3OutputModPresence::IfPresent) {
    if (ClampIdx < 0 && OmodIdx < 0)
      return true;

    int64_t Clamp = 0;
    int64_t Omod = 0;
    if ((ClampIdx >= 0 && !readNamedImm(Di, AMDGPU::OpName::clamp, Clamp)) ||
        (OmodIdx >= 0 && !readNamedImm(Di, AMDGPU::OpName::omod, Omod))) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " has malformed clamp/omod operands; operand table layout does "
           "not match the expected VOP3 profile")
              .str());
      return false;
    }

    if (Clamp != 0 || Omod != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " with non-default clamp/omod is not yet lifted; output modifier "
           "semantics must not be silently dropped")
              .str());
      return false;
    }
    return true;
  }

  // Required presence: both operands must exist and be immediates.
  if (Diag == VOP3OutputModDiag::FpValuSplit) {
    std::optional<int64_t> Clamp =
        readNamedImmOperand(Di, AMDGPU::OpName::clamp);
    if (!Clamp) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " missing immediate clamp operand; operand table layout does not "
           "match the expected VOP3 profile")
              .str());
      return false;
    }
    if (*Clamp != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " has clamp=1; VOP3 floating-point output clamp is not modeled "
           "for this opcode")
              .str());
      return false;
    }

    std::optional<int64_t> Omod =
        readNamedImmOperand(Di, AMDGPU::OpName::omod);
    if (!Omod) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " missing immediate omod operand; operand table layout does not "
           "match the expected VOP3 profile")
              .str());
      return false;
    }
    if (*Omod != 0) {
      Hr.Failure = RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3",
          (Twine(DiagnosticName) +
           " has nonzero omod; VOP3 floating-point output scaling is not "
           "modeled for this opcode")
              .str());
      return false;
    }
    return true;
  }

  int64_t Clamp = 0;
  int64_t Omod = 0;
  if (!readNamedImm(Di, AMDGPU::OpName::clamp, Clamp) ||
      !readNamedImm(Di, AMDGPU::OpName::omod, Omod)) {
    const char *MissingSuffix =
        Diag == VOP3OutputModDiag::PseudoScalar
            ? " missing immediate clamp/omod operands; operand table layout "
              "does not match the gfx12 VOP3 pseudo-scalar profile"
            : " missing immediate clamp/omod operands; operand table layout "
              "does not match the expected VOP3 profile";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3", (Twine(DiagnosticName) + MissingSuffix).str());
    return false;
  }

  if (Clamp != 0 || Omod != 0) {
    const char *RefusalSuffix =
        Diag == VOP3OutputModDiag::PseudoScalar
            ? " with non-default clamp/omod is not yet lifted; the base "
              "instruction is supported through an AMDGPU hardware "
              "intrinsic, but output modifier semantics must not be "
              "silently dropped"
            : " with non-default clamp/omod is not yet lifted; output "
              "modifier semantics must not be silently dropped";
    Hr.Failure = RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3", (Twine(DiagnosticName) + RefusalSuffix).str());
    return false;
  }

  return true;
}

} // namespace COMGR::hotswap
