//===-- include/flang/Runtime/OpenMP/omp_alloc.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_OMP_ALLOC_H_
#define FORTRAN_RUNTIME_OMP_ALLOC_H_

#include "flang/Runtime/descriptor-consts.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime::omp {

extern "C" {
void RTDECL(OpenMPRegisterAllocator)();
void RTDECL(OpenMPAllocatableSetAllocIdx)(Descriptor &descriptor, int pos);
}

} // namespace Fortran::runtime::omp
#endif // FORTRAN_RUNTIME_OMP_ALLOC_H_
