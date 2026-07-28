# Hotswap subproject coordinator. Each hotswap path is built as an OBJECT
# library whose TUs land directly in amd_comgr.so:
#   hotswap::rewriter          byte-level rewrite path (linked unconditionally)
#   hotswap::raiser            IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
#   hotswap::translation_cache on-disk cache for transpiled code objects
# Shared, path-agnostic headers live in common/ (header-only, no target).
add_subdirectory(rewriter)
add_subdirectory(raiser)
add_subdirectory(cache)
