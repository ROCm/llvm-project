# GTest discovery for Comgr unit tests. Three sources, in priority order:
#
# 1. In-tree llvm_gtest target (LLVM monorepo build with utils/unittest/).
# 2. LLVM-installed gtest (-DLLVM_INSTALL_GTEST=ON on the LLVM build),
#    discovered via find_library in LLVM_LIBRARY_DIRS with NO_DEFAULT_PATH.
#    Preferred over find_package(GTest) for standalone Comgr builds against
#    an installed ROCm/LLVM, because find_package(GTest CONFIG) searches
#    CMake's default paths and can pick up a system gtest at /usr/local
#    that wasn't built with -fPIC, breaking PIE link of the test binaries
#    (https://github.com/ROCm/llvm-project/issues/2505).
# 3. Vanilla GTest CMake package (find_package(GTest CONFIG)). Used by
#    superproject builds (e.g. TheRock) that supply googletest via the
#    standard GTest:: imported targets. This avoids forcing the LLVM
#    build to enable LLVM_INSTALL_GTEST just to test Comgr.
#
# If none is available, skip the test-unit suite with a warning rather
# than failing configure.
if(TARGET llvm_gtest)
  set(COMGR_GTEST_LIBS llvm_gtest_main llvm_gtest)
  set(COMGR_GTEST_INCLUDE_DIRS "")
else()
  find_library(COMGR_LLVM_GTEST_LIB llvm_gtest
    PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(COMGR_LLVM_GTEST_MAIN_LIB llvm_gtest_main
    PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_path(COMGR_LLVM_GTEST_INCLUDE_DIR gtest/gtest.h
    PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm-gtest NO_DEFAULT_PATH)
  if(COMGR_LLVM_GTEST_LIB AND COMGR_LLVM_GTEST_MAIN_LIB
      AND COMGR_LLVM_GTEST_INCLUDE_DIR)
    set(COMGR_GTEST_LIBS
      ${COMGR_LLVM_GTEST_MAIN_LIB} ${COMGR_LLVM_GTEST_LIB})
    set(COMGR_GTEST_INCLUDE_DIRS ${COMGR_LLVM_GTEST_INCLUDE_DIR})
  else()
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
      set(COMGR_GTEST_LIBS GTest::gtest_main GTest::gtest)
      set(COMGR_GTEST_INCLUDE_DIRS "")
    else()
      message(WARNING
        "Comgr test-unit skipped: no llvm_gtest target, no LLVM-installed "
        "gtest at ${LLVM_INCLUDE_DIRS}/llvm-gtest, and no GTest CMake "
        "package found. Provide one of: in-tree LLVM with utils/unittest/, "
        "-DLLVM_INSTALL_GTEST=ON on the LLVM build, or find_package(GTest).")
      return()
    endif()
  endif()
endif()

# Pin C++17 and mirror LLVM's RTTI setting (avoids type_info link errors
# against LLVM libs built with RTTI off, including hotswap::raiser
# which carries its own non-RTTI / non-exceptions compile flags via
# llvm_update_compile_flags()).
function(comgr_configure_test_target target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED Yes
    CXX_EXTENSIONS No)
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(${target} PRIVATE /GR-)
    else()
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
  endif()
endfunction()

# Hotswap unit tests live in their own subtree mirroring src/hotswap:
# hotswap/rewriter (byte-level rewrite path) and hotswap/raiser (IR transpiler).
# The gtest discovery above and comgr_configure_test_target() are inherited by
# the subdirectories.
add_subdirectory(hotswap)

# Register every test binary with the test-unit / check-comgr plumbing.
add_custom_target(test-unit
  COMMAND $<TARGET_FILE:HotswapElfTests>
  COMMAND $<TARGET_FILE:HotswapMCTests>
  COMMAND $<TARGET_FILE:HotswapProfileTests>
  COMMAND $<TARGET_FILE:RaiserScaffoldingTests>)
add_dependencies(test-unit HotswapElfTests HotswapMCTests HotswapProfileTests
  RaiserScaffoldingTests)
add_dependencies(check-comgr test-unit)
