set(LLVM_LINK_COMPONENTS
    Support
)

add_clang_tool(amdllvm
    amdllvm.cpp
)

list(APPEND CLANG_LINKS_TO_CREATE
  clang clang++ clang-cl clang-cpp clang-${CLANG_VERSION_MAJOR} lld)
# amdflang -> amdllvm (re-execs sibling flang). Omit when Flang is not in
# this install tree (e.g. standalone amd-llvm-flang owns amdflang instead).
option(CLANG_ENABLE_AMDFLANG "Install amdflang as an amdllvm symlink" ON)
if(CLANG_ENABLE_AMDFLANG)
  list(APPEND CLANG_LINKS_TO_CREATE flang)
endif()

foreach(link ${CLANG_LINKS_TO_CREATE})
    add_clang_symlink("amd${link}" amdllvm)
endforeach()
