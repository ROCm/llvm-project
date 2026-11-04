file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependency.c"
  "int dependency(void) { return 42; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/root.c"
  "int dependency(void); int root(void) { return dependency(); }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.c"
  "#if defined(PRIVATE_REQUIREMENT) && !defined(EXPECT_PUBLIC)\n"
  "#error Private compile requirement leaked\n#endif\n"
  "#if defined(EXPECT_PUBLIC) && !defined(PRIVATE_REQUIREMENT)\n"
  "#error Public compile requirement missing\n#endif\n"
  "int root(void); int main(void) { return root() != 42; }\n")
foreach(kind IN ITEMS STATIC SHARED)
  add_library(private-${kind} ${kind} "${CMAKE_CURRENT_BINARY_DIR}/dependency.c")
  target_compile_definitions(private-${kind} INTERFACE PRIVATE_REQUIREMENT)
  add_library(private-alias-${kind} ALIAS private-${kind})
  add_library(private-interface-${kind} INTERFACE)
  target_link_libraries(private-interface-${kind} INTERFACE private-alias-${kind})
  add_library(root-${kind} STATIC "${CMAKE_CURRENT_BINARY_DIR}/root.c")
  target_link_libraries(root-${kind} PRIVATE private-interface-${kind})
  foreach(visibility IN ITEMS private public repeated)
    set(consumer usage-${kind}-${visibility})
    add_executable(${consumer} "${CMAKE_CURRENT_BINARY_DIR}/main.c")
    if(visibility STREQUAL "public")
      target_compile_definitions(${consumer} PRIVATE EXPECT_PUBLIC)
      comgr_static_link_libraries(${consumer} root-${kind} private-interface-${kind})
    else()
      comgr_static_link_libraries(${consumer} root-${kind})
    endif()
    if(visibility STREQUAL "repeated")
      # A later public path must contribute requirements even after a private
      # path visited the same target, including across separate helper calls.
      target_compile_definitions(${consumer} PRIVATE EXPECT_PUBLIC)
      comgr_static_link_libraries(${consumer} private-interface-${kind})
    endif()
    add_test(NAME run-${consumer} COMMAND ${consumer})
  endforeach()
endforeach()
