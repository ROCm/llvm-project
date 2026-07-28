//===- pipeline.h - Transpile pipeline result -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PipelineResult is the transpile pipeline's output: the lowered HSACO plus
// the attribution fields the translation cache persists and restores. Only
// the result type is declared here -- the pipeline itself is not part of the
// translation-cache module.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_PIPELINE_H
#define HOTSWAP_TRANSPILER_PIPELINE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <memory>
#include <string>

namespace COMGR::hotswap {

struct PipelineTimings {
  double totalSeconds = 0.0;
  double listKernelsSeconds = 0.0;
  double extractTextSeconds = 0.0;
  double createTempDirSeconds = 0.0;
  double raiseSeconds = 0.0;
  double writeIrSeconds = 0.0;
  double optSeconds = 0.0;
  double llcSeconds = 0.0;
  double linkSeconds = 0.0;
  double readHsacoSeconds = 0.0;
  double collectMetadataSeconds = 0.0;
};

struct PipelineResult {
  std::unique_ptr<llvm::MemoryBuffer> Hsaco;
  PipelineTimings Timings;
  std::string FailMnemonic;
  std::string FailKernel;
  std::string FailReason;
  std::string FailFormat;
  std::string FailDetail;
  uint64_t FailOffset = 0;
  // Successful raises can still carry proof-relevant attribution. Today this
  // records C5 predicate-chain sites accepted under a projection-specific
  // proof (for example single-source-wave MODREP with no active replica
  // lanes); loader proof logs surface these fields on `hotswap_result`.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool TargetEnablePrivateSegment = false;
  uint32_t TargetPrivateSegmentFixedSize = 0;
  int LiftedCount = 0;
  int TotalCount = 0;
  // ScaledModuloReplicationProjection requirement for the (single) transpiled
  // kernel: the factor (W_t/W_s) the launch runtime must scale the block's x
  // extent by. 1 means no scaling.
  unsigned ScaledDispatchFactor = 1;
  bool Success = false;
};

} // namespace COMGR::hotswap

#endif
