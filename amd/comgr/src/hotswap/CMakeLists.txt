# Hotswap subproject coordinator. Each path is built as an OBJECT library whose
# TUs land directly in amd_comgr.so:
#   hotswap::rewriter  byte-level rewrite path (linked unconditionally)
#   hotswap::raiser    IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
#   hotswap::loader    code-object metadata loader feeding the raiser (opt-in)
#   hotswap::common    small shared pieces the transpile path links (opt-in)
# common/ is mostly header-only (kernel-meta.h); hotswap::common compiles the
# one out-of-line HotswapError::ID symbol the loader (and later decoder) share.
#
# All four keep their LLVM dependencies PRIVATE. Their objects land in
# amd_comgr, which already links the full curated LLVM set; propagating a
# partial set onto amd_comgr's link line perturbs static-archive resolution
# (dropping LLVMCodeGen's -mcpu registration and breaking in-process LLD) and
# mixes component archives with libLLVM.so under LLVM_LINK_LLVM_DYLIB. Targets
# that link these libraries directly name their own LLVM libs.
add_subdirectory(rewriter)
add_subdirectory(raiser)

# The loader consumes AMDGPU target-private headers an install-tree-only build
# does not expose, so the transpile-path libraries are entered only when the
# transpile entry point is on.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_subdirectory(common)
  add_subdirectory(loader)
endif()
