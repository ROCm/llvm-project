set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitReader
  BitWriter
  Core
  Support
  TransformUtils
  )

add_llvm_unittest(BitcodeTests
  BitReaderTest.cpp
  DataLayoutUpgradeTest.cpp
  SummaryExtensionTest.cpp
  )
