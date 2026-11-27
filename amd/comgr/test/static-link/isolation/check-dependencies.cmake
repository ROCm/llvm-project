execute_process(COMMAND "${READELF}" -d "${LIBRARY}"
  RESULT_VARIABLE status OUTPUT_VARIABLE dynamic ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect ${LIBRARY}: ${error}")
endif()
if(dynamic MATCHES "NEEDED[^\n]*(LLVM|LTO|Remarks)")
  message(FATAL_ERROR "Compiler shared library in ${LIBRARY}: ${dynamic}")
endif()
