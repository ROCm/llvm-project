# Hotswap subproject coordinator. Both hotswap paths are built as OBJECT
# libraries whose TUs land directly in amd_comgr.so:
#   hotswap::rewriter  byte-level rewrite path (linked unconditionally)
#   hotswap::raiser    IR transpiler path (opt-in, COMGR_ENABLE_HOTSWAP_TRANSPILE)
# Shared, path-agnostic headers live in common/ (header-only, no target).
add_subdirectory(rewriter)
add_subdirectory(raiser)
