set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  LTO
  Support
  )

add_llvm_unittest(LTOTests
  LTOCacheKeyTest.cpp
  )
