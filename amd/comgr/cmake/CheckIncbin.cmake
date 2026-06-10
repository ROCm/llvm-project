function(check_comgr_incbin_support output_var)
  set(_have_incbin_support OFF)

  if(DEFINED COMGR_CHECK_INCBIN_BINARY_DIR)
    set(_binary_dir "${COMGR_CHECK_INCBIN_BINARY_DIR}")
  else()
    set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  file(MAKE_DIRECTORY "${_binary_dir}")

  # Create a tiny assembly snippet that uses .incbin.
  set(_test_asm_file "${_binary_dir}/test_incbin.s")
  set(_test_asm_object "${_binary_dir}/test_incbin.o")
  set(_test_asm_source "
    .p2align 12
    .global incbin_test
incbin_test:
    .incbin \"${_test_asm_file}\"
    .byte 0
")

  file(WRITE "${_test_asm_file}" "${_test_asm_source}")
  file(REMOVE "${_test_asm_object}")

  if(CMAKE_ASM_COMPILER)
    execute_process(
      COMMAND ${CMAKE_ASM_COMPILER}
        -c "${_test_asm_file}" -o "${_test_asm_object}"
      RESULT_VARIABLE _asm_result)

    if(_asm_result EQUAL 0 AND EXISTS "${_test_asm_object}")
      set(_have_incbin_support ON)
    endif()
  endif()

  set(${output_var} ${_have_incbin_support} PARENT_SCOPE)
endfunction()
