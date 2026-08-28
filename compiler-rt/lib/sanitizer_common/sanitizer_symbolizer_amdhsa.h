//===-- sanitizer_symbolizer_amdhsa.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef SANITIZER_SYMBOLIZER_AMDHSA_H
#define SANITIZER_SYMBOLIZER_AMDHSA_H

#include "sanitizer_platform.h"

#if SANITIZER_AMDHSA
#  include "sanitizer_common.h"
#  include "sanitizer_symbolizer_comgr.h"

namespace __sanitizer {

// Symbolizer for AMDGPU CodeObject.
class AMDGPUCodeObjectSymbolizer {
 public:
  AMDGPUCodeObjectSymbolizer() : object_cnt(comgr_objects::no_objs) {}

  void Init(int fd, u64 offset, u64 size);
  bool SymbolizePC(uptr addr, InternalScopedString& source_loc);
  void Release();

 private:
  void InitCOMgr();
  amd_comgr_data_t codeobject;
  amd_comgr_symbolizer_info_t symbolizer;
  enum comgr_objects { no_objs = 0, data = 1, data_and_symb = 2 } object_cnt;
  bool init = false;
};
}  // namespace __sanitizer
#endif  // SANITIZER_AMDHSA
#endif  // SANITIZER_SYMBOLIZER_AMDHSA_H
