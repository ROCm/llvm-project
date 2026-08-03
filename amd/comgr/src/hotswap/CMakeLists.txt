# Hotswap subproject coordinator. Each hotswap path is built as an OBJECT
# library whose TUs land directly in amd_comgr.so:
#   hotswap::rewriter          byte-level rewrite path (linked unconditionally)
#   hotswap::raiser            IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
#   hotswap::translation_cache on-disk cache for transpiled code objects
# Shared, path-agnostic headers live in common/ (header-only, no target).
#
# Both keep their LLVM dependencies PRIVATE. Their objects land in amd_comgr,
# which already links the full curated LLVM set; propagating a partial set onto
# amd_comgr's link line perturbs static-archive resolution (dropping
# LLVMCodeGen's -mcpu registration and breaking in-process LLD) and mixes
# component archives with libLLVM.so under LLVM_LINK_LLVM_DYLIB. Targets that
# link these libraries directly name their own LLVM libs.
add_subdirectory(rewriter)
add_subdirectory(raiser)
add_subdirectory(cache)
