function(canonicalize_cmake_boolean var)
    if(${var})
        set(${var} 1 PARENT_SCOPE)
    else()
        set(${var} 0 PARENT_SCOPE)
    endif()
endfunction()

canonicalize_cmake_boolean(COMGR_SPIRV_BACKEND_AVAILABLE)
canonicalize_cmake_boolean(COMGR_SPIRV_TRANSLATOR_AVAILABLE)
canonicalize_cmake_boolean(COMGR_ENABLE_HOTSWAP_TRANSPILE)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

if (NOT DEFINED LLVM_LIT_PATH)
  find_program(LLVM_LIT_PATH
    NAMES llvm-lit llvm-lit.py llvm-lit.cmd
    PATHS "${LLVM_TOOLS_BINARY_DIR}/../../bin" "${LLVM_TOOLS_BINARY_DIR}"
    NO_DEFAULT_PATH
  )
  if (NOT LLVM_LIT_PATH)
    set(LLVM_LIT_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
  endif()
endif()
message("-- LLVM_LIT_PATH: ${LLVM_LIT_PATH}")

add_custom_target(test-lit COMMAND "${LLVM_LIT_PATH}"
                  "${CMAKE_CURRENT_BINARY_DIR}" -v)
add_dependencies(test-lit amd_comgr)

macro(add_comgr_lit_binary name lang)
  add_executable("${name}" "comgr-sources/${name}.${lang}")
  if (${lang} STREQUAL "c")
    set_target_properties("${name}" PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED Yes
      C_EXTENSIONS No)
  endif()
  target_link_libraries("${name}" amd_comgr)
  add_dependencies(check-comgr "${name}")
endmacro()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_comgr_lit_binary(source-to-bc-with-dev-libs c)
add_comgr_lit_binary(spirv-translator c)
add_comgr_lit_binary(compile-opencl-minimal c)
add_comgr_lit_binary(compile-hip-minimal c)
add_comgr_lit_binary(compile-hip-asan c)
add_comgr_lit_binary(spirv-to-reloc c)
add_comgr_lit_binary(source-to-spirv c)
add_comgr_lit_binary(unbundle c)
add_comgr_lit_binary(get-version c)
add_comgr_lit_binary(status-string c)
add_comgr_lit_binary(data-action c)
add_comgr_lit_binary(lookup-code-object c)
add_comgr_lit_binary(hotswap-rewrite c)
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)

  # raise_cli is the per-kernel transpile-to-IR driver used by the
  # hotswap-transpile/ lit fixtures. It links the in-tree hotswap
  # transpiler OBJECT library directly (the public amd_comgr API only
  # exposes the rewriting flow, not raw IR emission), so it doesn't fit
  # the add_comgr_lit_binary macro that links amd_comgr. Also compiles
  # in comgr-metadata.cpp because hotswap calls into
  # `metadata::walkElfMetadataIntoDoc` / `getElfIsaNameFromBuffer`;
  # `--gc-sections` drops the unreferenced bits.
  add_executable(raise_cli
    comgr-sources/raise_cli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-symbol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../test-shared/standalone-init.cpp)
  llvm_update_compile_flags(raise_cli)
  # Mirror amd_comgr's RTTI/EH posture: every typeinfo we touch comes
  # from LLVM libs that were built with -fno-rtti, so any TU here that
  # emits typeinfo (Error/ErrorInfo, format_adapter, etc.) will fail
  # to link if compiled with RTTI on.
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(raise_cli PRIVATE /GR-)
    else()
      target_compile_options(raise_cli PRIVATE -fno-rtti)
    endif()
  endif()
  target_include_directories(raise_cli PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../include)
  # `amd_comgr` provides the public C entry points
  # (`amd_comgr_create_data` / `set_data` / `release_data`) that
  # `code-object-utils.cpp` calls into to wrap raw ELF bytes for the
  # `metadata::getMetadataRoot` / `metadata::getElfIsaName` paths.
  target_link_libraries(raise_cli PRIVATE hotswap::transpiler amd_comgr)
  if(NOT MSVC)
    target_compile_options(raise_cli PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(raise_cli PRIVATE -Wl,--gc-sections)
  endif()
  add_dependencies(check-comgr raise_cli)
endif()

add_dependencies(check-comgr test-lit)
