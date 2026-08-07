add_subdirectory(rewriter)

# The raiser and decoder unit tests link the transpile-path OBJECT libraries,
# entered only under COMGR_ENABLE_HOTSWAP_TRANSPILE; skip them otherwise.
if(COMGR_ENABLE_HOTSWAP_TRANSPILE)
  add_subdirectory(raiser)
  add_subdirectory(decoder)
  list(APPEND COMGR_HOTSWAP_TEST_TARGETS
    RaiserScaffoldingTests WaveProjectionTests DecoderTests)
endif()

# The test binaries above, exported to the parent scope so the top-level
# test-unit target runs them without naming each one. Adding a hotswap test
# target updates this list, not test-unit/CMakeLists.txt.
set(COMGR_HOTSWAP_TEST_TARGETS "${COMGR_HOTSWAP_TEST_TARGETS}" PARENT_SCOPE)
