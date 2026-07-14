// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hiptensor.h"
#include "cutensor.h"
// CHECK-NOT: #include "hiptensor.h"

int main() {
  printf("25.before.20000. cuTensor API to hipTensor API synthetic test\n");

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType;
  cudaDataType dataType;
#endif

#if CUTENSOR_MAJOR < 2
  // CHECK: hiptensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_NONE = HIPTENSOR_AUTOTUNE_MODE_NONE;
  // CHECK-NEXT hiptensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_INCREMENTAL = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
  cutensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_NONE = CUTENSOR_AUTOTUNE_NONE;
  cutensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_INCREMENTAL = CUTENSOR_AUTOTUNE_INCREMENTAL;
#endif

  return 0;
}
