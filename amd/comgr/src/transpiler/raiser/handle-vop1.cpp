//===- handle-vop1.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/raiser/handle-vop-shared.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/raise_failure.h"

#include "llvm/Support/Error.h"

using namespace llvm;

namespace COMGR::transpiler {

Error handleVOP1(RaiseContext &Ctx, const DecodedInst &Di,
                 OperandResolver &Op) {
  switch (Di.CanonOp) {
  case CanonicalOp::V_MOV_B32:
    return raiseMove32(Ctx, Di, Op);
  case CanonicalOp::V_MOV_B64:
    return raiseMove64(Ctx, Di, Op);
  case CanonicalOp::V_NOT_B32:
  case CanonicalOp::V_BFREV_B32:
  case CanonicalOp::V_FFBH_U32:
  case CanonicalOp::V_FFBL_B32:
  case CanonicalOp::V_FFBH_I32:
    return raiseUnaryBit32(Ctx, Di, Op);
  default:
    return unsupportedInstruction(Ctx, Di);
  }
}

} // namespace COMGR::transpiler
