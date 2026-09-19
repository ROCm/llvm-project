file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/isolated.c"
  "#include <expected.h>\n"
  "#if COMGR_INTERFACE_REQUIREMENT != EXPECTED\n#error Missing usage requirements\n#endif\n"
  "int entry(void); int shared(void);\n"
  "int isolated(void) { return entry() == EXPECTED && shared() == EXPECTED; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.c"
  "int isolated(void); int main(void) { return !isolated(); }\n")

foreach(kind IN ITEMS imported builtin)
  if(kind STREQUAL "imported")
    set(root imported_first)
  else()
    set(root first)
    # An LLVM external-project build uses the global component list.
    set_property(GLOBAL PROPERTY LLVM_LIBS ${LLVM_AVAILABLE_LIBS})
    unset(LLVM_AVAILABLE_LIBS)
    comgr_static_llvm_libraries(llvm_libraries)
  endif()
  add_library(isolated-${kind} SHARED "${CMAKE_CURRENT_BINARY_DIR}/isolated.c")
  comgr_static_link_libraries(isolated-${kind} ${root} shared ${llvm_libraries})
  # Keep accidental shared dependencies visible even if no symbol needs them.
  target_link_options(isolated-${kind} PRIVATE -Wl,--no-undefined -Wl,--no-as-needed)
  add_executable(run-${kind} "${CMAKE_CURRENT_BINARY_DIR}/main.c")
  target_link_libraries(run-${kind} PRIVATE isolated-${kind})
  add_test(NAME run-isolated-${kind} COMMAND run-${kind})
  add_test(NAME dependencies-${kind}
    COMMAND "${CMAKE_COMMAND}" "-DREADELF=${CMAKE_READELF}"
      "-DLIBRARY=$<TARGET_FILE:isolated-${kind}>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check-dependencies.cmake")
endforeach()
