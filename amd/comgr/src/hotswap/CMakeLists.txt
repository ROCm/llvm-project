# Hotswap subproject coordinator. Each path is built as an OBJECT library whose
# TUs land directly in amd_comgr.so:
#   hotswap::rewriter  byte-level rewrite path (linked unconditionally)
#   hotswap::raiser    IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
#   hotswap::loader    code-object metadata loader feeding the raiser (opt-in)
#   hotswap::common    small shared pieces the transpile path links (opt-in)
# common/ is mostly header-only (kernel-meta.h); hotswap::common compiles the
# one out-of-line HotswapError::ID symbol the loader (and later decoder) share.
add_subdirectory(rewriter)
add_subdirectory(raiser)

# The loader consumes AMDGPU target-private headers an install-tree-only build
# does not expose, so the transpile-path libraries are entered only when the
# transpile entry point is on.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_subdirectory(common)
  add_subdirectory(loader)
endif()
