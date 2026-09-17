//===- SQTTFuncMap.cpp - SQTT funcmap emission ----------------------------===//
//
// Part of AMD SQTT Marker, under the MIT License. See
// amd/sqtt-marker/LICENSE.txt for license information.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Emits marker metadata into the .sqtt_funcmap code-object section.
///
//===----------------------------------------------------------------------===//

#include "SQTTPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

void SQTTInstrumentPass::emitFuncMap(Module &M) {
  // Bucket once, preserving insertion order within every non-function row
  // class. Only functions need an ID sort.
  auto RowFor = [](MarkerKind Kind) {
    if (Kind == MarkerKind::Function)
      return 0u;
    if (Kind == MarkerKind::Kernel)
      return 1u;
    if (Kind == MarkerKind::UserScope || Kind == MarkerKind::Point)
      return 2u;
    if (Kind == MarkerKind::SystemPoint)
      return 3u;
    if (Kind == MarkerKind::AddressPoint)
      return 4u;
    llvm_unreachable("invalid marker kind");
  };
  SmallVector<const MarkerRecord *, 8> Rows[5];
  for (const MarkerRecord &Entry : Markers)
    Rows[RowFor(Entry.Kind)].push_back(&Entry);
  llvm::sort(Rows[0], [](const MarkerRecord *A, const MarkerRecord *B) {
    return A->ID < B->ID;
  });

  std::string MapData;
  if (ShaderClockBitsUsed > 0) {
    MapData += "M:shader_clock_bits=";
    MapData += std::to_string(ShaderClockBitsUsed);
    MapData += ";shader_clock_shift=";
    MapData += std::to_string(Config.ShaderClockShift);
    MapData += '\n';
  }

  for (unsigned Row = 0; Row < 5; ++Row) {
    if (Row == 4 && AddrTraceWaveSize > 0) {
      MapData += "W:";
      MapData += std::to_string(AddrTraceWaveSize);
      MapData += '\n';
    }
    for (const MarkerRecord *Entry : Rows[Row]) {
      char Kind = Entry->Kind == MarkerKind::Kernel      ? 'K'
                  : Entry->Kind == MarkerKind::Function  ? 'F'
                  : Entry->Kind == MarkerKind::UserScope ? 'U'
                                                         : 'P';
      MapData += Kind;
      MapData += ':';
      if (Kind != 'K') {
        MapData += std::to_string(Entry->ID);
        MapData += ':';
      }
      MapData += Entry->Name;
      if (!Entry->SourceLoc.empty()) {
        MapData += '@';
        MapData += Entry->SourceLoc;
      }
      MapData += '\n';
      if (Kind != 'K' && Entry->ExtraPayloadCount) {
        MapData += "R:";
        MapData += std::to_string(Entry->ID);
        MapData += ":extra_payload_count=";
        MapData += std::to_string(Entry->ExtraPayloadCount);
        MapData += '\n';
      }
    }
  }

  LLVMContext &Ctx = M.getContext();
  Constant *StrConst = ConstantDataArray::getString(Ctx, MapData,
                                                    /*AddNull=*/true);

  // Use addrspace(1) for AMDGPU global memory
  unsigned AS = M.getDataLayout().getDefaultGlobalsAddressSpace();
  GlobalVariable *GV = new GlobalVariable(
      M, StrConst->getType(),
      /*isConstant=*/true, GlobalValue::InternalLinkage, StrConst,
      ".sqtt_func_id_map",
      /*InsertBefore=*/nullptr, GlobalVariable::NotThreadLocal, AS);
  GV->setSection(".sqtt_funcmap");
  GV->setAlignment(Align(1));

  appendToUsed(M, {GV});
}
