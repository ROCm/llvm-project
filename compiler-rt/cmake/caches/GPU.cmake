# --! DELETE ME AFFTER ROCK BUILD SCRIPTS ARE UPDATED !---
set(COMPILER_RT_INCLUDE_TESTS ON CACHE BOOL "")
set(COMPILER_RT_HAS_SAFESTACK OFF CACHE BOOL "")

set(COMPILER_RT_BUILD_BUILTINS ON CACHE BOOL "")
set(COMPILER_RT_BAREMETAL_BUILD ON CACHE BOOL "")
set(COMPILER_RT_BUILD_CRT OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_SANITIZERS ON CACHE BOOL "")
set(COMPILER_RT_SANITIZERS_TO_BUILD "ubsan_minimal" CACHE STRING "")
set(COMPILER_RT_BUILD_XRAY OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
# Build the device profile runtime (libclang_rt.profile.a) for the GPU target.
# With LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON (set by the amdgcn-amd-amdhsa
# runtimes target), this installs to lib/clang/<v>/lib/amdgcn-amd-amdhsa/ where
# the HIP/AMDGPU device toolchain resolves it on a -fprofile-instr-generate /
# -fcoverage-mapping device link. Without it, the device link fails with
# "undefined symbol: __llvm_profile_instrument_gpu".
set(COMPILER_RT_BUILD_PROFILE ON CACHE BOOL "")
# This is a freestanding device build (-nostdlibinc, no host libc headers), so
# the profile runtime must use its baremetal subset: it drops the filesystem /
# value-profiling sources (InstrProfilingFile/Util/GCDA/Runtime/Value) and makes
# InstrProfilingPort.h skip <unistd.h>. The device-side instrumentation
# (InstrProfilingPlatformGPU.c: __llvm_profile_instrument_gpu + the
# __llvm_profile_sections bounds table) is kept. Without this the build fails
# with "'unistd.h'/'fcntl.h'/'sys/file.h' file not found".
set(COMPILER_RT_PROFILE_BAREMETAL ON CACHE BOOL "")
# The host-side HIP drain (InstrProfilingPlatformROCm.cpp) dlopen's HSA/HIP and
# is host-only; it must never be compiled for the amdgcn device target. The
# device archive only needs the instrumentation runtime (InstrProfilingPlatformGPU.c).
set(COMPILER_RT_BUILD_PROFILE_ROCM OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_MEMPROF OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_XRAY_NO_PREINIT OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_ORC OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_GWP_ASAN OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_SCUDO_SANTDALONE_WITH_LLVM_LIBC OFF CACHE BOOL "")
