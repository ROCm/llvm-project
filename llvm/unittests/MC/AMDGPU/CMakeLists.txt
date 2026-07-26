include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUDisassembler
  AMDGPUInfo
  MC
  MCDisassembler
  Support
  TargetParser
  )

add_llvm_unittest(AMDGPUMCTests
  Disassembler.cpp
  DwarfRegMappings.cpp
  PARTIAL_SOURCES_INTENDED
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUDisassembler
  AMDGPUInfo
  AMDGPUUtils
  MC
  MCDisassembler
  Support
  TargetParser
  )

add_llvm_target_unittest(AMDGPUOperandInfoTests
  OperandInfo.cpp
  PARTIAL_SOURCES_INTENDED
  )
