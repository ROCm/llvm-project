set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  IPO
  LTO
  Support
  )

add_llvm_unittest(LTOTests
  LTOBackendTest.cpp
  LTOCacheKeyTest.cpp
  )

target_link_libraries(LTOTests PRIVATE LLVMTestingSupport)
