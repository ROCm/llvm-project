add_llvm_component_library(LLVMX86Desc
  X86ATTInstPrinter.cpp
  X86IntelInstPrinter.cpp
  X86InstComments.cpp
  X86InstPrinterCommon.cpp
  X86EncodingOptimization.cpp
  X86ShuffleDecode.cpp
  X86AsmBackend.cpp
  X86MCTargetDesc.cpp
  X86MCAsmInfo.cpp
  X86MCCodeEmitter.cpp
  X86MachObjectWriter.cpp
  X86MnemonicTables.cpp
  X86ELFObjectWriter.cpp
  X86WinCOFFObjectWriter.cpp
  X86WinCOFFStreamer.cpp
  X86WinCOFFTargetStreamer.cpp

  LINK_COMPONENTS
  BinaryFormat
  CodeGenTypes
  MC
  MCDisassembler
  Support
  TargetParser
  X86Info

  ADD_TO_COMPONENT
  X86
  )
set_source_files_properties(X86MCCodeEmitter.cpp PROPERTIES COMPILE_FLAGS "-O2")
