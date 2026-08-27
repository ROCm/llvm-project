function(canonicalize_cmake_boolean var)
    if(${var})
        set(${var} 1 PARENT_SCOPE)
    else()
        set(${var} 0 PARENT_SCOPE)
    endif()
endfunction()

canonicalize_cmake_boolean(COMGR_SPIRV_BACKEND_AVAILABLE)
canonicalize_cmake_boolean(COMGR_SPIRV_TRANSLATOR_AVAILABLE)

if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
    set(COMGR_AMDGPU_TARGET_AVAILABLE 1)
else()
    set(COMGR_AMDGPU_TARGET_AVAILABLE 0)
endif()
canonicalize_cmake_boolean(COMGR_AMDGPU_TARGET_AVAILABLE)

canonicalize_cmake_boolean(COMGR_ENABLE_HOTSWAP_TRANSPILE)

find_program(LLVM_LIT_PATH
  NAMES llvm-lit llvm-lit.py llvm-lit.cmd
  PATHS "${LLVM_TOOLS_BINARY_DIR}/../../bin" "${LLVM_TOOLS_BINARY_DIR}"
  NO_DEFAULT_PATH
)
if (NOT LLVM_LIT_PATH)
  set(LLVM_LIT_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
endif()
get_filename_component(COMGR_LIT_LLVM_TOOLS_DIR "${LLVM_LIT_PATH}" DIRECTORY)
message("-- LLVM_LIT_PATH: ${LLVM_LIT_PATH}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(test-lit COMMAND "${LLVM_LIT_PATH}"
                  "${CMAKE_CURRENT_BINARY_DIR}" -v)

macro(add_comgr_lit_binary name lang)
  add_executable("${name}" "comgr-sources/${name}.${lang}")
  if (${lang} STREQUAL "c")
    set_target_properties("${name}" PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED Yes
      C_EXTENSIONS No)
  endif()
  target_link_libraries("${name}" amd_comgr)
  add_dependencies(test-lit "${name}")
endmacro()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_comgr_lit_binary(source-to-bc-with-dev-libs c)
add_comgr_lit_binary(spirv-translator c)
add_comgr_lit_binary(compile-opencl-minimal c)
add_comgr_lit_binary(logger-redirect c)
add_comgr_lit_binary(compile-hip-minimal c)
add_comgr_lit_binary(compile-hip-asan c)
add_comgr_lit_binary(compile-hip-cxx-headers c)
add_comgr_lit_binary(spirv-to-reloc c)
add_comgr_lit_binary(source-to-spirv c)
add_comgr_lit_binary(unbundle c)
add_comgr_lit_binary(unpackage c)
add_comgr_lit_binary(get-version c)
add_comgr_lit_binary(status-string c)
add_comgr_lit_binary(data-action c)
add_comgr_lit_binary(lookup-code-object c)
add_comgr_lit_binary(hotswap-rewrite c)
add_comgr_lit_binary(parse-isa-name c)
add_comgr_lit_binary(name-expression-map c)
add_comgr_lit_binary(isa-enumeration c)
add_comgr_lit_binary(test-get-data-isa-name c)

# The hotswap transpiler driver used by the hotswap/raiser lit tests. It links
# the hotswap decoder, raiser, loader, and common OBJECT libraries and the
# shared comgr-object-utils objects (which carry the buffer-facing
# getElfIsaName / getMetadataRoot / lookupSymbolByName entry points) rather than
# recompiling the comgr sources, so the driver and amd_comgr share one set of
# objects and compile flags. Those objects also contain amd_comgr C-API helpers
# the driver never calls and that reference the rest of comgr; --gc-sections
# drops them (they were built with function/data sections). MSVC's default
# /OPT:REF removes them the same way.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_executable(hotswap_transpile_cli comgr-sources/hotswap_transpile_cli.cpp)
  # Match the C++ standard the rest of comgr compiles with. The comgr build
  # sets it per target rather than globally, and MSVC otherwise defaults below
  # the C++17 that the LLVM headers this driver includes require.
  set_target_properties(hotswap_transpile_cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS No)
  # Pick up LLVM's exception posture for the host compiler.
  llvm_update_compile_flags(hotswap_transpile_cli)
  # LLVM is built without RTTI; match it explicitly so cl::opt's OptionValue
  # typeinfo is not emitted against the -fno-rtti Support library
  # (llvm_update_compile_flags does not add this in a standalone comgr build).
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(hotswap_transpile_cli PRIVATE /GR-)
    else()
      target_compile_options(hotswap_transpile_cli PRIVATE -fno-rtti)
    endif()
  endif()

  target_include_directories(hotswap_transpile_cli PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../include)
  # The hotswap OBJECT libraries contribute objects only, so name the LLVM set
  # this driver needs explicitly; ${LLVM_LIBS} is the same curated set amd_comgr
  # links, already resolved to the dylib or the static components. It carries
  # the AMDGPU components the driver's LLVMInitializeAMDGPU* calls register.
  target_link_libraries(hotswap_transpile_cli
    PRIVATE hotswap::common hotswap::loader hotswap::decoder hotswap::raiser
            comgr-object-utils ${LLVM_LIBS})
  if(NOT WIN32)
    target_link_options(hotswap_transpile_cli PRIVATE -Wl,--gc-sections)
  endif()
  add_dependencies(test-lit hotswap_transpile_cli)
endif()

add_dependencies(check-comgr test-lit)
