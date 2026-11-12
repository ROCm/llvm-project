include(CheckCXXSourceCompiles)

set(embed_test_code "
static const unsigned char data[] = {
#embed <CMakeLists.txt>
};
int main() { return data[0]; }
")

check_cxx_source_compiles("${embed_test_code}" HAVE_EMBED_SUPPORT)

if(HAVE_EMBED_SUPPORT)
  message(STATUS "Compiler supports #embed directive in C++")
else()
  message(STATUS "Compiler does NOT support #embed directive in C++")
endif()


# Create a tiny assembly snippet that uses .incbin
set(TEST_ASM_SOURCE "
    .p2align 12
    .global incbin_test
incbin_test:
    .incbin \"${CMAKE_CURRENT_BINARY_DIR}/test_incbin.s\"
    .byte 0
")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_incbin.s ${TEST_ASM_SOURCE})

set(HAVE_INCBIN_SUPPORT OFF)
set(TEST_ASM_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/test_incbin.o)
file(REMOVE ${TEST_ASM_OBJECT})

if(CMAKE_ASM_COMPILER)
  execute_process(
    COMMAND ${CMAKE_ASM_COMPILER} -c ${CMAKE_CURRENT_BINARY_DIR}/test_incbin.s
                                  -o ${TEST_ASM_OBJECT}
    RESULT_VARIABLE asm_result)

  if(asm_result EQUAL 0 AND EXISTS ${TEST_ASM_OBJECT})
    set(HAVE_INCBIN_SUPPORT ON)
  endif()
endif()

if(HAVE_INCBIN_SUPPORT)
  message(STATUS "Assembler supports .incbin directive")
else()
  message(STATUS "Assembler does NOT support .incbin directive")
endif()
