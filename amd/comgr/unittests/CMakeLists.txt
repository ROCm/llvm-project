set(LLVM_LINK_COMPONENTS
  Object
  Support
  TargetParser
  )

add_unittest(check-comgr HotswapElfTests
  HotswapElfTest.cpp
  ../src/comgr-hotswap-elf.cpp)

target_include_directories(HotswapElfTests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)
