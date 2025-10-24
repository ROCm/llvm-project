# For upstream, we accumulate all libraries into the MLIR_CAPI_LIBS
# property via a custom wrapper function. This is then used to create an
# aggregate below.
set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBS)
function(add_mlir_upstream_c_api_library name)
  add_mlir_public_c_api_library(${name} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBS ${name})
endfunction()

add_subdirectory_optional_install(Debug)
add_subdirectory_optional_install(Dialect)
add_subdirectory_optional_install(Conversion)
add_subdirectory_optional_install(Interfaces)
add_subdirectory_optional_install(IR)
add_subdirectory_optional_install(RegisterEverything)
add_subdirectory_optional_install(Transforms)
add_subdirectory_optional_install(Target)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_subdirectory_optional_install(ExecutionEngine)
endif()

# Build the optional CAPI dylib.
if(MLIR_BUILD_MLIR_C_DYLIB)
  message(STATUS "Building MLIR-C dylib")
  get_property(_capi_libraries GLOBAL PROPERTY MLIR_CAPI_LIBS)
  add_mlir_aggregate(MLIR-C
    SHARED
    EMBED_LIBS
      ${_capi_libraries}
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(MLIR-C PRIVATE "-Wl,-exclude-libs,ALL")
  else()
    if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
      message(STATUS "MLIR-C on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
    endif()
  endif()
endif()
