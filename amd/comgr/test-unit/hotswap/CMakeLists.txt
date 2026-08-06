add_subdirectory(rewriter)
add_subdirectory(raiser)

# The decoder unit tests link the hotswap::decoder library, which only exists
# when COMGR_ENABLE_HOTSWAP_TRANSPILE is on; skip them otherwise.
if(TARGET hotswap::decoder)
  add_subdirectory(decoder)
endif()
