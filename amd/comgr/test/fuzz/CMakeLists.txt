add_llvm_fuzzer(HotswapRewriteFuzzer
  HotswapRewriteFuzzer.cpp
  DUMMY_MAIN DummyHotswapRewriteFuzzer.cpp)

if(TARGET HotswapRewriteFuzzer)
  target_link_libraries(HotswapRewriteFuzzer PRIVATE amd_comgr)
  set_target_properties(HotswapRewriteFuzzer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED Yes
    CXX_EXTENSIONS No)

  set(HOTSWAP_FUZZER_SEED
    "${CMAKE_CURRENT_BINARY_DIR}/Inputs/hotswap-rewrite.hsaco")
  add_custom_command(
    OUTPUT "${HOTSWAP_FUZZER_SEED}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/Inputs"
    COMMAND "$<TARGET_FILE:clang>" -target amdgcn-amd-amdhsa -mcpu=gfx1250
            -nostdlib
            "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/hotswap-rewrite.s"
            -o "${HOTSWAP_FUZZER_SEED}"
    DEPENDS clang Inputs/hotswap-rewrite.s
    VERBATIM)
  add_custom_target(HotswapRewriteFuzzerSeed
    DEPENDS "${HOTSWAP_FUZZER_SEED}")
  add_dependencies(HotswapRewriteFuzzer HotswapRewriteFuzzerSeed)
  add_dependencies(check-comgr HotswapRewriteFuzzer)

  # The dummy main runs explicit files once. A libFuzzer-enabled build uses
  # its own corpus-directory command line and is exercised separately.
  if(NOT LLVM_LIB_FUZZING_ENGINE AND NOT LLVM_USE_SANITIZE_COVERAGE)
    add_test(NAME comgr_hotswap_rewrite_fuzzer_smoke
      COMMAND HotswapRewriteFuzzer
              "${HOTSWAP_FUZZER_SEED}"
              "${CMAKE_CURRENT_SOURCE_DIR}/Inputs/hotswap-rewrite.s")
    if(UNIX)
      set_tests_properties(comgr_hotswap_rewrite_fuzzer_smoke PROPERTIES
        ENVIRONMENT "AMD_COMGR_CACHE=0;")
    else()
      set_tests_properties(comgr_hotswap_rewrite_fuzzer_smoke PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_LINKER_FILE_DIR:amd_comgr>;AMD_COMGR_CACHE=set:0")
    endif()
  endif()
endif()
