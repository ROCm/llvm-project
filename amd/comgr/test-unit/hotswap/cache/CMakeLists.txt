# Unit tests for the hotswap translation cache (hotswap/cache/translation-cache.h).
#
# Pins the on-disk content-addressed cache contract: first-run miss then hit,
# key sensitivity to source hash / ISA / opt-level / kernel name, and the
# invalid/readonly/skip-list paths. The test builds its own fake AMDGPU ELF,
# so it needs no GPU and no real code object.

add_executable(TranslationCacheTests
  TranslationCacheTest.cpp)

target_link_libraries(TranslationCacheTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::translation_cache
  ${LLVM_PTHREAD_LIB})

target_include_directories(TranslationCacheTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(TranslationCacheTests)

add_test(NAME TranslationCacheTests
  COMMAND $<TARGET_FILE:TranslationCacheTests>)

add_executable(MemCacheTests
  MemCacheTest.cpp)

target_compile_definitions(MemCacheTests PRIVATE
  COMGR_HOTSWAP_MEM_CACHE_TESTING)

target_link_libraries(MemCacheTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::translation_cache
  ${LLVM_PTHREAD_LIB})

target_include_directories(MemCacheTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(MemCacheTests)

add_test(NAME MemCacheTests
  COMMAND $<TARGET_FILE:MemCacheTests>)
