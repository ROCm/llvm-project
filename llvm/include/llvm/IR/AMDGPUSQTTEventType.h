//===- AMDGPUSQTTEventType.h - AMDGPU SQTT event types ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Event kinds for the \c llvm.amdgcn.sqtt.event intrinsic. The values are part
/// of the IR operand format and are checked by the verifier.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_AMDGPUSQTTEVENTTYPE_H
#define LLVM_IR_AMDGPUSQTTEVENTTYPE_H

namespace llvm {
namespace sqtt {

enum class EventType {
  FunctionEntry,
  FunctionExit,
  UserEntry,
  UserExit,
  SetEventTableBase,
  End,
  Begin = FunctionEntry,
};

} // namespace sqtt
} // namespace llvm

#endif
