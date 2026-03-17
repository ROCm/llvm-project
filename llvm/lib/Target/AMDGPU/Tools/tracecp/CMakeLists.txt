include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include(AddLLVM)

add_executable(tracecp
  tracecp.cpp
  TraceUtil.cpp
)

llvm_update_compile_flags(tracecp)

target_link_libraries(tracecp PRIVATE
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUDisassembler
  LLVMAMDGPUInfo
  LLVMAMDGPUSim
  LLVMAMDGPUSimMCAdapter
  LLVMMC
  LLVMMCDisassembler
  LLVMSupport
  LLVMTargetParser
)

set_target_properties(tracecp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)
set_target_properties(tracecp PROPERTIES EXCLUDE_FROM_ALL TRUE)
