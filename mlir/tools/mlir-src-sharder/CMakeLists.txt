set(LLVM_LINK_COMPONENTS Support)

# add_tablegen links LLVM statically, but MLIRSupport links the LLVM dylib when
# LLVM_LINK_LLVM_DYLIB is set; on Windows that duplicates every LLVMSupport
# symbol. Build the one MLIRSupport file this tool needs into it instead.
if(LLVM_LINK_LLVM_DYLIB)
  set(LIBS)
  set(_sharder_extra_sources ${MLIR_MAIN_SRC_DIR}/lib/Support/FileUtilities.cpp)
else()
  set(LIBS MLIRSupport)
  set(_sharder_extra_sources)
endif()

add_tablegen(mlir-src-sharder MLIR_SRC_SHARDER
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  mlir-src-sharder.cpp
  ${_sharder_extra_sources})

set_target_properties(mlir-src-sharder PROPERTIES FOLDER "MLIR/Tablegenning")
target_link_libraries(mlir-src-sharder PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-src-sharder)
