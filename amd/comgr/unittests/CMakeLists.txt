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
  $<TARGET_OBJECTS:comgr_hotswap_obj>)

target_include_directories(HotswapTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
