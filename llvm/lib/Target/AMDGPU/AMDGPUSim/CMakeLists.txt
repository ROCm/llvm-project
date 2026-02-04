#Core simulation library(no CodeGen / MC dependencies)
add_llvm_component_library(LLVMAMDGPUSim HWModel.cpp Simulator.cpp

                               PARTIAL_SOURCES_INTENDED

                                   LINK_COMPONENTS Support

                                       ADD_TO_COMPONENT AMDGPU)

#MIR adapter(depends on CodeGen)
    add_llvm_component_library(LLVMAMDGPUSimMIRAdapter MIRAdapter.cpp

                                   PARTIAL_SOURCES_INTENDED

                                       LINK_COMPONENTS AMDGPUSim CodeGen Support

                                           ADD_TO_COMPONENT AMDGPU)

#MC adapter(depends on MC)
        add_llvm_component_library(LLVMAMDGPUSimMCAdapter MCAdapter.cpp

                                       PARTIAL_SOURCES_INTENDED

                                           LINK_COMPONENTS AMDGPUSim MC Support

                                               ADD_TO_COMPONENT AMDGPU)
