##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.ll
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../irif/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../oclc/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gaaf.cl
  PROPERTIES COMPILE_FLAGS "-munsafe-fp-atomics")

opencl_bc_lib(NAME ockl SOURCES ${sources})
