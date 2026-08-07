add_subdirectory(rewriter)

# The raiser and decoder unit tests link the transpile-path OBJECT libraries,
# entered only under COMGR_ENABLE_HOTSWAP_TRANSPILE; skip them otherwise.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  # The transpile OBJECT libraries keep their LLVM dependencies PRIVATE, so a
  # test executable names its own. libLLVM does not export the AMDGPU
  # target-private helpers the decoder and wave projection call (mc2PseudoReg,
  # getNamedOperandIdx, the isGFXn predicates), so link the static AMDGPU
  # components rather than the dylib; a static-plus-dylib link would also
  # re-register the AMDGPU cl::opts and abort at startup.
  llvm_map_components_to_libnames(COMGR_HOTSWAP_TEST_LLVM_LIBS
    AMDGPU
    Analysis AsmPrinter BinaryFormat CodeGen Core MC MCDisassembler
    MCParser Object Passes Support Target TargetParser TransformUtils)

  add_subdirectory(raiser)
  add_subdirectory(decoder)
  list(APPEND COMGR_HOTSWAP_TEST_TARGETS
    RaiserScaffoldingTests WaveProjectionTests RaiseFailureTests RegFileTests
    KernargAbiLayoutTests DecoderTests)
endif()

# The test binaries above, exported to the parent scope so the top-level
# test-unit target runs them without naming each one. Adding a hotswap test
# target updates this list, not test-unit/CMakeLists.txt.
set(COMGR_HOTSWAP_TEST_TARGETS "${COMGR_HOTSWAP_TEST_TARGETS}" PARENT_SCOPE)
