# hotswap-tblgen: build-time generator for the raiser's canonical opcode
# tables. See CanonicalOpcodeEmitter.cpp for what it emits and why.
#
# This is a TableGen tool in the `clang-tblgen` / `mlir-tblgen` mould: a stock
# TableGen driver plus one Comgr-specific backend. It links the TableGen
# support libraries (all three are exported by LLVMExports.cmake) and needs
# two source-tree include paths that the LLVM install tree does not carry:
#
#   utils/TableGen      Common/CodeGenTarget.h, Common/CodeGenInstruction.h
#   lib/Target/AMDGPU   SIDefines.h (the SIInstrFlags TSFlags masks)
#
# The parent CMakeLists.txt already hard-requires an LLVM *build* tree for the
# same class of reason, so this adds no new configuration mode -- but it does
# additionally require utils/, so check for it explicitly rather than failing
# later with a confusing missing-header error.
if(NOT EXISTS "${LLVM_BUILD_MAIN_SRC_DIR}/utils/TableGen/Common/CodeGenTarget.h")
  message(FATAL_ERROR
    "hotswap-tblgen needs the LLVM TableGen backend support headers at "
    "${LLVM_BUILD_MAIN_SRC_DIR}/utils/TableGen/Common/, but they are not "
    "there. Point LLVM_DIR at a build tree whose source tree includes "
    "utils/TableGen (a full llvm-project checkout).")
endif()

set(HOTSWAP_TBLGEN_SOURCES
  HotswapTableGen.cpp
  CanonicalOpcodeEmitter.cpp
)

# `add_tablegen` (llvm/cmake/modules/TableGen.cmake) is preferred because it
# handles cross-compilation: with LLVM_USE_HOST_TOOLS it arranges a host-native
# build of the tool and points HOTSWAP_TABLEGEN_EXE at it. It is available once
# `include(TableGen)` has run, which the parent does. Fall back to a plain
# executable when it is not (older LLVM CMake packages); that is correct for
# native builds, which is all the fallback needs to cover.
#
# No DESTINATION: this is an internal build tool and is not installed.
if(COMMAND add_tablegen)
  # External projects such as Comgr call project() in their own directory,
  # which changes PROJECT_NAME from LLVM to amd_comgr (and hotswap changes it
  # once more). LLVM's build_native_tool() keys the already-configured native
  # build directory and CONFIGURE_*_NATIVE target off PROJECT_NAME, however.
  # Temporarily restore LLVM while add_tablegen wires its host tool so builds
  # with LLVM_OPTIMIZED_TABLEGEN or cross-compilation reuse LLVM's native
  # build instead of looking for an unconfigured hotswap-transpiler native
  # build at /bin.
  set(_hotswap_tablegen_project_name "${PROJECT_NAME}")
  if(LLVM_USE_HOST_TOOLS AND TARGET CONFIGURE_LLVM_NATIVE)
    set(PROJECT_NAME LLVM)
  endif()
  add_tablegen(hotswap-tblgen HOTSWAP ${HOTSWAP_TBLGEN_SOURCES})
  set(PROJECT_NAME "${_hotswap_tablegen_project_name}")
  unset(_hotswap_tablegen_project_name)
else()
  add_executable(hotswap-tblgen ${HOTSWAP_TBLGEN_SOURCES})
  set(HOTSWAP_TABLEGEN_EXE hotswap-tblgen PARENT_SCOPE)
  set(HOTSWAP_TABLEGEN_TARGET hotswap-tblgen PARENT_SCOPE)
endif()

target_include_directories(hotswap-tblgen PRIVATE
  ${LLVM_INCLUDE_DIRS}
  "${LLVM_BUILD_MAIN_SRC_DIR}/utils/TableGen"
  "${LLVM_BUILD_MAIN_SRC_DIR}/lib/Target/AMDGPU"
)

target_link_libraries(hotswap-tblgen PRIVATE
  LLVMTableGenCommon
  LLVMTableGenBasic
  LLVMTableGen
  LLVMCodeGenTypes
  LLVMSupport
  LLVMDemangle
)

llvm_update_compile_flags(hotswap-tblgen)
