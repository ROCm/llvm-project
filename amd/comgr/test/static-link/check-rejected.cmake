execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${SOURCE}" -B "${BINARY}"
    "-DCMAKE_C_COMPILER=${COMPILER}" "-DCOMGR_REJECT_TEST=${KIND}"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(KIND STREQUAL "conditional")
  set(expected "Unsupported static Comgr link dependency")
else()
  set(expected "Unterminated Comgr link expression")
endif()
if(status EQUAL 0 OR NOT error MATCHES "${expected}")
  message(FATAL_ERROR "Expected rejection of ${KIND}: ${output}${error}")
endif()
