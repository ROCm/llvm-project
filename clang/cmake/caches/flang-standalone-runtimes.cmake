# Cache for the Flang-standalone runtimes ExternalProject (-S runtimes).
# Builds flang-rt (intrinsic .mod files) + OpenMP Fortran modules only.

set(LLVM_ENABLE_RUNTIMES "flang-rt;openmp" CACHE STRING "")
set(LIBOMP_FORTRAN_MODULES_ONLY ON CACHE BOOL
  "Build omp_lib.mod without rebuilding libomp (provided by amd-llvm)")
set(RUNTIMES_FORTRAN_MODULES ON CACHE BOOL "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")

# Must agree with flang-standalone.cmake / the Flang driver.
set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath" CACHE STRING
  "Library implementing REAL(16) math for flang-rt")

# Skip smoke tests; Flang/flang-rt are still mid-build.
set(CMAKE_C_COMPILER_WORKS ON CACHE BOOL "")
set(CMAKE_CXX_COMPILER_WORKS ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER_WORKS ON CACHE BOOL "")
