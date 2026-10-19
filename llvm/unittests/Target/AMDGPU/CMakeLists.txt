include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  ${PROJECT_SOURCE_DIR}/unittests/CodeGen
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Analysis
  AsmParser
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  MIRParser
  Passes
  Support
  TargetParser
  )

add_llvm_target_unittest(AMDGPUTests
  AMDGPUMCExprTest.cpp
  AMDGPUUnitTests.cpp
  CSETest.cpp
  DwarfRegMappings.cpp
  ExecMayBeModifiedBeforeAnyUse.cpp
  GCNRegPressureTest.cpp
  InstSizes.cpp
  LiveRegUnits.cpp
  NextUseAnalysisIncrementalTest.cpp
  PALMetadata.cpp
  RCIUpdateReservedRegsTest.cpp
  UniformityAnalysisTest.cpp
  MFMACoExecRules.cpp
  )

target_compile_definitions(AMDGPUTests PRIVATE
  "LLVM_AMDGPU_NEXT_USE_MIR_DIR=\"${PROJECT_SOURCE_DIR}/test/CodeGen/AMDGPU/NextUseAnalysis\"")
