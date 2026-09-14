cmake_minimum_required(VERSION 3.13.4)
project(ComgrStaticLinkTest LANGUAGES C)
include(../../cmake/ComgrStaticLibraries.cmake)
enable_testing()

# The late archive refers back to a separate object in the first archive.
# Linking only the roots in a rescan group leaves this reference unresolved.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/entry.c"
  "int late(void); int entry(void) { return late(); }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/helper.c"
  "int helper(void) { return 42; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/late.c"
  "int helper(void); int late(void) { return helper(); }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/shared.c"
  "int shared(void) { return 42; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.c"
  "int entry(void); int shared(void); int main(void) { return entry() != shared(); }\n")
add_library(first STATIC "${CMAKE_CURRENT_BINARY_DIR}/entry.c"
  "${CMAKE_CURRENT_BINARY_DIR}/helper.c")
add_library(late STATIC "${CMAKE_CURRENT_BINARY_DIR}/late.c")
add_library(shared SHARED "${CMAKE_CURRENT_BINARY_DIR}/shared.c")

# Exercise LINK_ONLY, an alias, and an intervening interface library.
add_library(late_alias ALIAS late)
add_library(dependencies INTERFACE)
target_link_libraries(dependencies INTERFACE "$<LINK_ONLY:late_alias>")
target_link_libraries(first PRIVATE dependencies)
comgr_collect_static_libraries(libraries first shared)
add_executable(static-link "${CMAKE_CURRENT_BINARY_DIR}/main.c")
target_link_libraries(static-link PRIVATE
  "-Wl,--start-group" ${libraries} "-Wl,--end-group")
add_test(NAME run-static-link COMMAND static-link)

# Cycles must terminate, and separate calls must not share traversal state.
add_library(cycle_a INTERFACE)
add_library(cycle_b INTERFACE)
target_link_libraries(cycle_a INTERFACE cycle_b)
target_link_libraries(cycle_b INTERFACE cycle_a late)
comgr_collect_static_libraries(cycle_libraries cycle_a)
if(NOT "late" IN_LIST cycle_libraries)
  message(FATAL_ERROR "Missing archive reachable through an interface cycle")
endif()
comgr_collect_static_libraries(repeated_libraries first shared)
if(NOT "${libraries}" STREQUAL "${repeated_libraries}")
  message(FATAL_ERROR "Static library traversal leaked state between calls")
endif()
