# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

add_subdirectory_optional_install(Analysis)
add_subdirectory_optional_install(AsmParser)
add_subdirectory_optional_install(Bytecode)
add_subdirectory_optional_install(Conversion)
add_subdirectory_optional_install(Debug)
add_subdirectory_optional_install(Dialect)
add_subdirectory_optional_install(IR)
add_subdirectory_optional_install(Interfaces)
add_subdirectory_optional_install(Parser)
add_subdirectory_optional_install(Pass)
add_subdirectory_optional_install(Query)
add_subdirectory_optional_install(Reducer)
add_subdirectory_optional_install(Remark)
add_subdirectory_optional_install(Rewrite)
add_subdirectory_optional_install(Support)
add_subdirectory_optional_install(TableGen)
add_subdirectory_optional_install(Target)
add_subdirectory_optional_install(Tools)
add_subdirectory_optional_install(Transforms)
add_subdirectory_optional_install(ExecutionEngine)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_library(MLIRRegisterAllDialects
  RegisterAllDialects.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs}
  )

add_mlir_library(MLIRRegisterAllPasses
  RegisterAllPasses.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs} # Some passes are part of the dialect libs
  ${conversion_libs}
  )

add_mlir_library(MLIRRegisterAllExtensions
  RegisterAllExtensions.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  )
