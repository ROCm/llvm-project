message(STATUS, "GPURUN step2")

set(OPENMP_UTILS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")

macro(add_openmp_util path)
  install(PROGRAMS
     ${path}
     DESTINATION "${OPENMP_UTILS_INSTALL_DIR}")
endmacro()

message(STATUS, "GPURUN step3")

add_subdirectory(gpurun)
message(STATUS, "GPURUN step4")
