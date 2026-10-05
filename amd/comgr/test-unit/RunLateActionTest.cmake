# Prefer the rebuilt library while preserving execution-time dependency paths.
if(APPLE)
  set(LibraryPathVariable DYLD_LIBRARY_PATH)
elseif(UNIX)
  set(LibraryPathVariable LD_LIBRARY_PATH)
endif()

if(DEFINED LibraryPathVariable)
  set(LibraryPath "${COMGR_LIBRARY_DIR}")
  if(NOT "$ENV{${LibraryPathVariable}}" STREQUAL "")
    string(APPEND LibraryPath ":$ENV{${LibraryPathVariable}}")
  endif()
  set(ENV{${LibraryPathVariable}} "${LibraryPath}")
endif()

execute_process(COMMAND "${TEST_EXECUTABLE}" RESULT_VARIABLE Result)
if(NOT "${Result}" STREQUAL "0")
  message(FATAL_ERROR "LateActionTest failed: ${Result}")
endif()
