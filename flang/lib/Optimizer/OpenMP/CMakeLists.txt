get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FlangOpenMPTransforms
  DoConcurrentConversion.cpp
  FunctionFiltering.cpp
  GenericLoopConversion.cpp
  GlobalFiltering.cpp
  MapsForPrivatizedSymbols.cpp
  MapInfoFinalization.cpp
  MarkDeclareTarget.cpp
  LowerWorkshare.cpp
  LowerNontemporal.cpp

  DEPENDS
  FIRDialect
  HLFIROpsIncGen
  FlangOpenMPPassesIncGen

  LINK_LIBS
  FIRAnalysis
  FIRBuilder
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FortranSupport
  HLFIRDialect

  MLIR_DEPS
  ${dialect_libs}

  MLIR_LIBS
  MLIRFuncDialect
  MLIRMathTransforms
  MLIROpenMPDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ${dialect_libs}
)

add_subdirectory(Support)
