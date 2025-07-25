//===- GlobalHandler.h - Target independent global & environment handling -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent global handler and environment manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H

#include <type_traits>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"

#include "Shared/Debug.h"
#include "Shared/Utils.h"

#include "omptarget.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

class DeviceImageTy;
struct GenericDeviceTy;

using namespace llvm::object;

/// Common abstraction for globals that live on the host and device.
/// It simply encapsulates the symbol name, symbol size, and symbol address
/// (which might be host or device depending on the context).
struct GlobalTy {
  GlobalTy(const std::string &Name, uint32_t Size = 0, void *Ptr = nullptr)
      : Name(Name), Size(Size), Ptr(Ptr) {}

  const std::string &getName() const { return Name; }
  uint32_t getSize() const { return Size; }
  void *getPtr() const { return Ptr; }

  void setSize(int32_t S) { Size = S; }
  void setPtr(void *P) { Ptr = P; }

private:
  // NOTE: Maybe we can have a pointer to the offload entry name instead of
  // holding a private copy of the name as a std::string.
  std::string Name;
  uint32_t Size;
  void *Ptr;
};

using IntPtrT = void *;
struct __llvm_profile_data {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer)                     \
  std::remove_const<Type>::type Name;
#include "llvm/ProfileData/InstrProfData.inc"
};

extern "C" {
extern int __attribute__((weak)) __llvm_write_custom_profile(
    const char *Target, const __llvm_profile_data *DataBegin,
    const __llvm_profile_data *DataEnd, const char *CountersBegin,
    const char *CountersEnd, const char *NamesBegin, const char *NamesEnd,
    const uint64_t *VersionOverride);
}
/// PGO profiling data extracted from a GPU device
struct GPUProfGlobals {
  SmallVector<int64_t> Counts;
  SmallVector<__llvm_profile_data> Data;
  SmallVector<uint8_t> NamesData;
  Triple TargetTriple;
  uint64_t Version = INSTR_PROF_RAW_VERSION;

  void dump() const;
  Error write() const;
  bool empty() const;
};

/// Subclass of GlobalTy that holds the memory for a global of \p Ty.
template <typename Ty> struct StaticGlobalTy : public GlobalTy {
  template <typename... Args>
  StaticGlobalTy(const std::string &Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  template <typename... Args>
  StaticGlobalTy(const char *Name, Args &&...args)
      : GlobalTy(Name, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  template <typename... Args>
  StaticGlobalTy(const char *Name, const char *Suffix, Args &&...args)
      : GlobalTy(std::string(Name) + Suffix, sizeof(Ty), &Data),
        Data(Ty{std::forward<Args>(args)...}) {}

  Ty &getValue() { return Data; }
  const Ty &getValue() const { return Data; }
  void setValue(const Ty &V) { Data = V; }

private:
  Ty Data;
};

/// Helper class to do the heavy lifting when it comes to moving globals between
/// host and device. Through the GenericDeviceTy we access memcpy DtoH and HtoD,
/// which means the only things specialized by the subclass is the retrieval of
/// global metadata (size, addr) from the device.
/// \see getGlobalMetadataFromDevice
class GenericGlobalHandlerTy {
  /// Actually move memory between host and device. See readGlobalFromDevice and
  /// writeGlobalToDevice for the interface description.
  Error moveGlobalBetweenDeviceAndHost(GenericDeviceTy &Device,
                                       DeviceImageTy &Image,
                                       const GlobalTy &HostGlobal,
                                       bool Device2Host);

  /// Actually move memory between host and device. See readGlobalFromDevice and
  /// writeGlobalToDevice for the interface description.
  Error moveGlobalBetweenDeviceAndHost(GenericDeviceTy &Device,
                                       const GlobalTy &HostGlobal,
                                       const GlobalTy &DeviceGlobal,
                                       bool Device2Host);

public:
  virtual ~GenericGlobalHandlerTy() {}

  /// Helper function for getting an ELF from a device image.
  Expected<std::unique_ptr<ObjectFile>> getELFObjectFile(DeviceImageTy &Image);

  /// Returns whether the symbol named \p SymName is present in the given \p
  /// Image.
  bool isSymbolInImage(GenericDeviceTy &Device, DeviceImageTy &Image,
                       StringRef SymName);

  /// Get the address and size of a global in the image. Address is
  /// returned in \p ImageGlobal and the global name is passed in \p
  /// ImageGlobal. If no size is present in \p ImageGlobal, then the size of the
  /// global will be stored there. If it is present, it will be validated
  /// against the real size of the global.
  Error getGlobalMetadataFromImage(GenericDeviceTy &Device,
                                   DeviceImageTy &Image, GlobalTy &ImageGlobal);

  /// Read the memory associated with a global from the image and store it on
  /// the host. The name, size, and destination are defined by \p HostGlobal.
  Error readGlobalFromImage(GenericDeviceTy &Device, DeviceImageTy &Image,
                            const GlobalTy &HostGlobal);

  /// Get the address and size of a global from the device. Address is
  /// returned in \p ImageGlobal and the global name is passed in \p
  /// ImageGlobal. If no size is present in \p ImageGlobal, then the size of the
  /// global will be stored there. If it is present, it will be validated
  /// against the real size of the global.
  virtual Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                            DeviceImageTy &Image,
                                            GlobalTy &DeviceGlobal) = 0;

  /// Copy the memory associated with a global from the device to its
  /// counterpart on the host. The name, size, and destination are defined by
  /// \p HostGlobal. The origin is defined by \p DeviceGlobal.
  Error readGlobalFromDevice(GenericDeviceTy &Device,
                             const GlobalTy &HostGlobal,
                             const GlobalTy &DeviceGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal, DeviceGlobal,
                                          /*D2H=*/true);
  }

  /// Copy the memory associated with a global from the device to its
  /// counterpart on the host. The name, size, and destination are defined by
  /// \p HostGlobal. The origin is automatically resolved.
  Error readGlobalFromDevice(GenericDeviceTy &Device, DeviceImageTy &Image,
                             const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, Image, HostGlobal,
                                          /*D2H=*/true);
  }

  /// Copy the memory associated with a global from the host to its counterpart
  /// on the device. The name, size, and origin are defined by \p HostGlobal.
  /// The destination is defined by \p DeviceGlobal.
  Error writeGlobalToDevice(GenericDeviceTy &Device, const GlobalTy &HostGlobal,
                            const GlobalTy &DeviceGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, HostGlobal, DeviceGlobal,
                                          /*D2H=*/false);
  }

  /// Copy the memory associated with a global from the host to its counterpart
  /// on the device. The name, size, and origin are defined by \p HostGlobal.
  /// The destination is automatically resolved.
  Error writeGlobalToDevice(GenericDeviceTy &Device, DeviceImageTy &Image,
                            const GlobalTy &HostGlobal) {
    return moveGlobalBetweenDeviceAndHost(Device, Image, HostGlobal,
                                          /*D2H=*/false);
  }

  /// Reads profiling data from a GPU image to supplied profdata struct.
  /// Iterates through the image symbol table and stores global values
  /// with profiling prefixes.
  Expected<GPUProfGlobals> readProfilingGlobals(GenericDeviceTy &Device,
                                                DeviceImageTy &Image);
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // LLVM_OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_GLOBALHANDLER_H
