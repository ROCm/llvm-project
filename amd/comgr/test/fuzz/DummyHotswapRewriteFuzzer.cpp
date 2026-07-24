//===- DummyHotswapRewriteFuzzer.cpp -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A file-input main for ordinary test builds that do not link libFuzzer.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(int Argc, char **Argv) {
  if (Argc < 2) {
    std::cerr << "usage: HotswapRewriteFuzzer <input> [<input> ...]\n";
    return 1;
  }

  for (int I = 1; I < Argc; ++I) {
    std::ifstream Input(Argv[I], std::ios::binary);
    if (!Input) {
      std::cerr << "could not open fuzz input: " << Argv[I] << '\n';
      return 1;
    }
    std::vector<uint8_t> Bytes{std::istreambuf_iterator<char>(Input),
                               std::istreambuf_iterator<char>()};
    if (Input.bad()) {
      std::cerr << "could not read fuzz input: " << Argv[I] << '\n';
      return 1;
    }
    LLVMFuzzerTestOneInput(Bytes.data(), Bytes.size());
  }
  return 0;
}
