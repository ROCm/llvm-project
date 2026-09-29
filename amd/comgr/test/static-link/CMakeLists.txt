cmake_minimum_required(VERSION 3.13.4)
project(ComgrStaticLinkTest LANGUAGES C)
include(../../cmake/ComgrStaticLibraries.cmake)
enable_testing()

if(COMGR_REJECT_TEST)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reject.c" "int main(void) { return 0; }\n")
  add_executable(reject "${CMAKE_CURRENT_BINARY_DIR}/reject.c")
  if(COMGR_REJECT_TEST STREQUAL "conditional")
    set(dependency "$<$<NOT:$<LINK_LANGUAGE:Swift>>:delayimp;-delayload:ole32.dll;LLVM>")
  else()
    set(dependency "$<$<NOT:$<LINK_LANGUAGE:Swift>>:delayimp;-delayload:ole32.dll")
  endif()
  comgr_static_link_libraries(reject "${dependency}")
  return()
endif()

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

# Model LLVM packages that export both component archives and shared wrappers.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llvm.c"
  "int external_only(void) { return 7; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/wrapper.c"
  "int external_only(void); int wrapper(void) { return external_only(); }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/other.c"
  "int external_only(void); int main(void) { return external_only() != 7; }\n")
add_library(LLVM SHARED "${CMAKE_CURRENT_BINARY_DIR}/llvm.c")
add_library(LTO SHARED "${CMAKE_CURRENT_BINARY_DIR}/wrapper.c")
add_library(Remarks SHARED "${CMAKE_CURRENT_BINARY_DIR}/wrapper.c")
# Avoid collisions with real LLVM libraries in the caller's loader path.
foreach(target IN ITEMS LLVM LTO Remarks)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME "comgr-test-${target}")
endforeach()
target_link_libraries(LTO PRIVATE LLVM)
target_link_libraries(Remarks PRIVATE LLVM)
target_link_libraries(first PRIVATE LLVM)
set_target_properties(first late PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Import in the parent directory, as a superproject can do before adding Comgr.
add_library(list-dependencies INTERFACE)
target_link_libraries(list-dependencies INTERFACE "$<LINK_ONLY:late_alias;LLVM>")
add_library(imported_first STATIC IMPORTED)
set_target_properties(imported_first PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}first${CMAKE_STATIC_LIBRARY_SUFFIX}"
  INTERFACE_LINK_LIBRARIES "dependencies;list-dependencies;$<LINK_ONLY:LLVM>")
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${config}" upper_config)
  set_property(TARGET imported_first PROPERTY "IMPORTED_LOCATION_${upper_config}"
    "${CMAKE_CURRENT_BINARY_DIR}/${config}/${CMAKE_STATIC_LIBRARY_PREFIX}first${CMAKE_STATIC_LIBRARY_SUFFIX}")
endforeach()
add_dependencies(imported_first first)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/expected.h" "#define EXPECTED 42\n")
foreach(target IN ITEMS first imported_first)
  set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_BINARY_DIR}")
  set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
    COMGR_INTERFACE_REQUIREMENT=42)
endforeach()

set(LLVM_AVAILABLE_LIBS first LLVM LTO Remarks)
comgr_static_llvm_libraries(llvm_libraries)
get_target_property(original_imported imported_first INTERFACE_LINK_LIBRARIES)
get_target_property(original_builtin first INTERFACE_LINK_LIBRARIES)
add_subdirectory(isolation)
get_target_property(current_imported imported_first INTERFACE_LINK_LIBRARIES)
get_target_property(current_builtin first INTERFACE_LINK_LIBRARIES)
if(NOT original_imported STREQUAL current_imported OR
   NOT original_builtin STREQUAL current_builtin)
  message(FATAL_ERROR "Comgr changed another consumer's link interface")
endif()

# An independent consumer must still inherit the shared LLVM dependency.
add_executable(other-consumer "${CMAKE_CURRENT_BINARY_DIR}/other.c")
target_link_libraries(other-consumer PRIVATE imported_first)
add_test(NAME run-other-consumer COMMAND other-consumer)

# Check the Windows LLVMSupport interface without invoking a Windows linker.
# EXCLUDE_FROM_ALL allows this configuration regression to run on Unix too.
# These flags defer loading system DLLs. Keep the fixtures independent of the
# recognizer so changes to its whitelist cannot mask bugs.
foreach(wrapper IN ITEMS plain intel reordered extended intel-extended)
  set(WL "")
  if(wrapper MATCHES "^intel")
    set(WL "$<$<OR:$<LINK_LANG_AND_ID:C,IntelLLVM>,$<LINK_LANG_AND_ID:CXX,IntelLLVM>,$<LINK_LANG_AND_ID:Fortran,IntelLLVM>>:-Qoption,link,>")
  endif()
  set(delayload "$<$<NOT:$<LINK_LANGUAGE:Swift>>:delayimp;${WL}-delayload:shell32.dll;${WL}-delayload:ole32.dll>")
  if(wrapper STREQUAL "reordered")
    set(delayload "$<$<NOT:$<LINK_LANGUAGE:Swift>>:-delayload:ole32.dll;delayimp;-delayload:shell32.dll>")
  elseif(wrapper MATCHES "extended$")
    set(delayload "$<$<NOT:$<LINK_LANGUAGE:Swift>>:delayimp;${WL}-delayload:ole32.dll;${WL}-delayload:shell32.dll;${WL}-delayload:advapi32.dll>")
  endif()
  add_library(windows-support-${wrapper} INTERFACE)
  target_link_libraries(windows-support-${wrapper} INTERFACE "${delayload}" LLVM)
  add_executable(windows-consumer-${wrapper} EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/main.c")
  comgr_static_link_libraries(windows-consumer-${wrapper} windows-support-${wrapper})
  get_target_property(windows_links windows-consumer-${wrapper} LINK_LIBRARIES)
  string(FIND "${windows_links}" "${delayload}" position)
  if(NOT position EQUAL 0)
    message(FATAL_ERROR "Windows delay-load expression was not preserved")
  endif()
  if("LLVM" IN_LIST windows_links)
    message(FATAL_ERROR "Windows consumer retained the LLVM dylib")
  endif()
endforeach()

foreach(kind IN ITEMS conditional unterminated)
  add_test(NAME reject-${kind}
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DBINARY=${CMAKE_CURRENT_BINARY_DIR}/reject-${kind}"
      "-DCOMPILER=${CMAKE_C_COMPILER}" "-DKIND=${kind}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check-rejected.cmake")
endforeach()
add_subdirectory(usage)

# Nested list expressions and literal brackets must leave the next item intact.
set(expected "$<LINK_ONLY:/tmp/lib<name.a>>")
set(pending "${expected};late")
comgr_pop_link_dependency(actual pending)
if(NOT actual STREQUAL expected OR NOT pending STREQUAL "late")
  message(FATAL_ERROR "Literal brackets consumed the next link item")
endif()
set(expected "$<$<BOOL:1>:$<LINK_ONLY:first>;late>")
set(pending "${expected};shared")
comgr_pop_link_dependency(actual pending)
if(NOT actual STREQUAL expected OR NOT pending STREQUAL "shared")
  message(FATAL_ERROR "Nested list expression was not reconstructed")
endif()

# Link a real archive whose filename contains literal brackets.
add_library(bracket STATIC "${CMAKE_CURRENT_BINARY_DIR}/shared.c")
set_target_properties(bracket PROPERTIES OUTPUT_NAME "bracket<name" SUFFIX ".a>")
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${config}" upper_config)
  set_property(TARGET bracket PROPERTY "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}"
    "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
add_executable(bracket-consumer "${CMAKE_CURRENT_BINARY_DIR}/main.c")
add_dependencies(bracket-consumer bracket)
comgr_static_link_libraries(bracket-consumer first
  "$<LINK_ONLY:${CMAKE_CURRENT_BINARY_DIR}/libbracket<name.a>>")
add_test(NAME run-bracket-consumer COMMAND bracket-consumer)
