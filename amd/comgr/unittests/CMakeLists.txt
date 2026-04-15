set(HOTSWAP_TEST_SOURCES
  ../src/comgr-hotswap-elf.cpp
  ../src/comgr-hotswap-llvm.cpp
  ../src/comgr-hotswap-b0a0.cpp)

set(LLVM_LINK_COMPONENTS
  MC
  MCDisassembler
  MCParser
  Object
  Support
  TargetParser
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  AllTargetsMCAs
  )

add_unittest(check-comgr HotswapTests
  HotswapTest.cpp
  ${HOTSWAP_TEST_SOURCES})

target_include_directories(HotswapTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
