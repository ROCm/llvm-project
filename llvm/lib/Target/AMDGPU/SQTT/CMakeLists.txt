add_llvm_component_library(LLVMAMDGPUSQTT
  Event.cpp

  LINK_COMPONENTS
  Core
  Support
  Object

  ADD_TO_COMPONENT
  AMDGPU
  )
