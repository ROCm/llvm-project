# Cache for the TheRock amd-llvm-flang subproject (standalone Flang + MLIR).
# Nested flang-rt / OpenMP module builds use flang-standalone-runtimes.cmake.

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_BUILD_TESTS OFF CACHE BOOL "")
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "")
set(FLANG_INCLUDE_TESTS OFF CACHE BOOL "")

# Must agree with flang-standalone-runtimes.cmake / flang-rt.
set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath" CACHE STRING
  "Library implementing REAL(16) math for flang-rt")

set(FLANG_STANDALONE_RUNTIMES_CACHE
  "${CMAKE_CURRENT_LIST_DIR}/flang-standalone-runtimes.cmake"
  CACHE FILEPATH
  "CMake -C cache for the flang-rt/openmp modules ExternalProject")
