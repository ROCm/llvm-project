if(NOT DEFINED COMGR_CMAKE_DIR)
  message(FATAL_ERROR "COMGR_CMAKE_DIR must be set")
endif()

if(NOT DEFINED COMGR_TEST_BINARY_DIR)
  message(FATAL_ERROR "COMGR_TEST_BINARY_DIR must be set")
endif()

include("${COMGR_CMAKE_DIR}/CheckIncbin.cmake")

file(REMOVE_RECURSE "${COMGR_TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${COMGR_TEST_BINARY_DIR}")

set(CMAKE_ASM_COMPILER "${CMAKE_COMMAND};-E;echo")
set(COMGR_CHECK_INCBIN_BINARY_DIR "${COMGR_TEST_BINARY_DIR}")
check_comgr_incbin_support(HAVE_INCBIN_SUPPORT)

if(HAVE_INCBIN_SUPPORT)
  message(FATAL_ERROR
    ".incbin check accepted an assembler that produced no object")
endif()
