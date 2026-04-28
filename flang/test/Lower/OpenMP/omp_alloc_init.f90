!RUN: %flang_fc1 -fopenmp -fopenmp-default-allocate=gpu -emit-fir %s -o - | FileCheck %s

program amd_alloc_init
    !CHECK: fir.call @_FortranAAMDRegisterAllocator()
end program amd_alloc_init