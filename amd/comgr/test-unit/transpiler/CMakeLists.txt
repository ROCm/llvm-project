# The raiser and decoder unit tests link the transpile-path OBJECT libraries,
# entered only under COMGR_ENABLE_TRANSPILER; skip them otherwise.
if(COMGR_ENABLE_TRANSPILER)
  # These tests require target-private AMDGPU helpers. Link only the static LLVM
  # components to avoid unresolved symbols and duplicate command-line option
  # registration.
  llvm_map_components_to_libnames(COMGR_TRANSPILER_TEST_LLVM_LIBS
    AMDGPU
    Analysis AsmPrinter BinaryFormat CodeGen Core MC MCDisassembler
    MCParser Object Passes Support Target TargetParser TransformUtils)

  # AMDGPU is registered into that LLVM by COMGR::ensureLLVMInitialized, whose
  # production definition lives inside amd_comgr; comgr-test-llvm-init defines
  # it for binaries that link the transpiler objects without linking amd_comgr.
  list(APPEND COMGR_TRANSPILER_TEST_LLVM_LIBS comgr-test-llvm-init)

  add_subdirectory(raiser)
  add_subdirectory(decoder)
  list(APPEND COMGR_TRANSPILER_TEST_TARGETS
    RaiserScaffoldingTests WaveProjectionTests RaiseFailureTests RegFileTests
    RegisterStateTests RaiseContextTests OperandResolverTests UserSgprLayoutTests
    DecoderTests)
endif()

# The test binaries above, exported to the parent scope so the top-level
# test-unit target runs them without naming each one. Adding a transpiler test
# target updates this list, not test-unit/CMakeLists.txt.
set(COMGR_TRANSPILER_TEST_TARGETS "${COMGR_TRANSPILER_TEST_TARGETS}" PARENT_SCOPE)
