include(CheckCXXSourceCompiles)
include("${CMAKE_CURRENT_LIST_DIR}/CheckIncbin.cmake")

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

check_comgr_incbin_support(HAVE_INCBIN_SUPPORT)

if(HAVE_INCBIN_SUPPORT)
  message(STATUS "Assembler supports .incbin directive")
else()
  message(STATUS "Assembler does NOT support .incbin directive")
endif()
