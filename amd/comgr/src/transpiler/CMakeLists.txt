# Transpiler subproject coordinator. Each path is built as an OBJECT library whose
# TUs land directly in amd_comgr.so:
#   transpiler::raiser    IR transpiler path (opt-in, COMGR_ENABLE_TRANSPILER)
#   transpiler::loader    code-object metadata loader feeding the raiser (opt-in)
#   transpiler::decoder   per-ISA AMDGPU MC stack + canonical-op identity (opt-in)
#   transpiler::common    small shared pieces the transpile path links (opt-in)
# common/ is mostly header-only (kernel-meta.h); transpiler::common compiles the
# one out-of-line TranspilerError::ID symbol the loader and decoder share.
#
# All four keep their LLVM dependencies PRIVATE. Their objects land in
# amd_comgr, which already links the full curated LLVM set; propagating a
# partial set onto amd_comgr's link line perturbs static-archive resolution
# (dropping LLVMCodeGen's -mcpu registration and breaking in-process LLD) and
# mixes component archives with libLLVM.so under LLVM_LINK_LLVM_DYLIB. Targets
# that link these libraries directly name their own LLVM libs.
#
# The raiser, loader, and decoder consume AMDGPU target-private headers an
# install-tree-only build does not expose, and the raiser links the decoder, so
# the transpile-path libraries are entered only when the transpile entry point
# is on.
if(COMGR_ENABLE_TRANSPILER)
  add_subdirectory(common)
  add_subdirectory(loader)
  add_subdirectory(decoder)
  add_subdirectory(raiser)
endif()

# The OBJECT libraries amd_comgr links, exported to the parent scope so the
# top-level CMakeLists links the set without naming each library. Empty unless
# COMGR_ENABLE_TRANSPILER is on. Adding a transpiler library updates this
# list, not the top-level CMakeLists.
set(COMGR_TRANSPILER_LIBS)
if(COMGR_ENABLE_TRANSPILER)
  list(APPEND COMGR_TRANSPILER_LIBS
    transpiler::common transpiler::loader transpiler::raiser transpiler::decoder)
endif()
set(COMGR_TRANSPILER_LIBS "${COMGR_TRANSPILER_LIBS}" PARENT_SCOPE)
