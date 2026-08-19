# Hotswap subproject coordinator. Each path is built as an OBJECT library whose
# TUs land directly in amd_comgr.so:
#   hotswap::rewriter  byte-level rewrite path (linked unconditionally)
#   hotswap::raiser    IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
#   hotswap::loader    code-object metadata loader feeding the raiser (opt-in)
#   hotswap::decoder   per-ISA AMDGPU MC stack + canonical-op identity (opt-in)
#   hotswap::common    small shared pieces the transpile path links (opt-in)
# common/ is mostly header-only (kernel-meta.h); hotswap::common compiles the
# one out-of-line HotswapError::ID symbol the loader and decoder share.
#
# All five keep their LLVM dependencies PRIVATE. Their objects land in
# amd_comgr, which already links the full curated LLVM set; propagating a
# partial set onto amd_comgr's link line perturbs static-archive resolution
# (dropping LLVMCodeGen's -mcpu registration and breaking in-process LLD) and
# mixes component archives with libLLVM.so under LLVM_LINK_LLVM_DYLIB. Targets
# that link these libraries directly name their own LLVM libs.
add_subdirectory(rewriter)

# The raiser, loader, and decoder consume AMDGPU target-private headers an
# install-tree-only build does not expose, and the raiser links the decoder, so
# the transpile-path libraries are entered only when the transpile entry point
# is on.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_subdirectory(common)
  add_subdirectory(loader)
  add_subdirectory(decoder)
  add_subdirectory(raiser)
endif()

# The OBJECT libraries amd_comgr links, exported to the parent scope so the
# top-level CMakeLists links the set without naming each library: the rewriter
# always, the transpile-path libraries only under COMGR_ENABLE_HOTSWAP_TRANSPILE.
# Adding a hotswap library updates this list, not the top-level CMakeLists.
set(COMGR_HOTSWAP_LIBS hotswap::rewriter)
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  list(APPEND COMGR_HOTSWAP_LIBS
    hotswap::common hotswap::loader hotswap::raiser hotswap::decoder)
endif()
# Created only when amd_comgr does not link LLVMAMDGPUUtils, and linked by
# amd_comgr alone: the AMDGPU MC tables it carries are a second definition
# anywhere the static AMDGPU components are on the link line.
if(TARGET hotswap::amdgpu-mc-tables)
  list(APPEND COMGR_HOTSWAP_LIBS hotswap::amdgpu-mc-tables)
endif()
set(COMGR_HOTSWAP_LIBS "${COMGR_HOTSWAP_LIBS}" PARENT_SCOPE)
