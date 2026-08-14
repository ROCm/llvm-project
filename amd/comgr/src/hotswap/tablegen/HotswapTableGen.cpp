//===- HotswapTableGen.cpp - Hotswap TableGen driver ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry point for `hotswap-tblgen`, the build-time generator for the hotswap
// raiser's canonical opcode tables.
//
// This is a stock TableGen driver (same shape as `llvm-tblgen`); the backends
// register themselves through `TableGen::Emitter::Opt` static initializers in
// the other translation units of this directory.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  MultiFileTableGenMainFn MainFn = nullptr;
  return TableGenMain(argv[0], MainFn);
}
