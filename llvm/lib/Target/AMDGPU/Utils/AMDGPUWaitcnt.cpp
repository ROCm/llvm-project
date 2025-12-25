//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUWaitcnt.h"
#include "AMDGPUBitMasks.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/TargetParser.h"

namespace {

/// \returns Vmcnt bit shift (lower bits).
unsigned getVmcntBitShiftLo(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 10 : 0;
}

/// \returns Vmcnt bit width (lower bits).
unsigned getVmcntBitWidthLo(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 6 : 4;
}

/// \returns Expcnt bit shift.
unsigned getExpcntBitShift(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 0 : 4;
}

/// \returns Expcnt bit width.
unsigned getExpcntBitWidth(unsigned VersionMajor) { return 3; }

/// \returns Lgkmcnt bit shift.
unsigned getLgkmcntBitShift(unsigned VersionMajor) {
  return VersionMajor >= 11 ? 4 : 8;
}

/// \returns Lgkmcnt bit width.
unsigned getLgkmcntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 10 ? 6 : 4;
}

/// \returns Vmcnt bit shift (higher bits).
unsigned getVmcntBitShiftHi(unsigned VersionMajor) { return 14; }

/// \returns Vmcnt bit width (higher bits).
unsigned getVmcntBitWidthHi(unsigned VersionMajor) {
  return (VersionMajor == 9 || VersionMajor == 10) ? 2 : 0;
}

/// \returns Loadcnt bit width
unsigned getLoadcntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 6 : 0;
}

/// \returns Samplecnt bit width.
unsigned getSamplecntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 6 : 0;
}

/// \returns Bvhcnt bit width.
unsigned getBvhcntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 3 : 0;
}

/// \returns Dscnt bit width.
unsigned getDscntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 6 : 0;
}

/// \returns Dscnt bit shift in combined S_WAIT instructions.
unsigned getDscntBitShift(unsigned VersionMajor) { return 0; }

/// \returns Storecnt or Vscnt bit width, depending on VersionMajor.
unsigned getStorecntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 10 ? 6 : 0;
}

/// \returns Kmcnt bit width.
unsigned getKmcntBitWidth(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 5 : 0;
}

/// \returns Xcnt bit width.
unsigned getXcntBitWidth(unsigned VersionMajor, unsigned VersionMinor) {
  return VersionMajor == 12 && VersionMinor == 5 ? 6 : 0;
}

/// \returns shift for Loadcnt/Storecnt in combined S_WAIT instructions.
unsigned getLoadcntStorecntBitShift(unsigned VersionMajor) {
  return VersionMajor >= 12 ? 8 : 0;
}

} // end anonymous namespace

namespace llvm {
namespace AMDGPU {

unsigned getVmcntBitMask(const IsaVersion &Version) {
  return (1 << (getVmcntBitWidthLo(Version.Major) +
                getVmcntBitWidthHi(Version.Major))) -
         1;
}

unsigned getLoadcntBitMask(const IsaVersion &Version) {
  return (1 << getLoadcntBitWidth(Version.Major)) - 1;
}

unsigned getSamplecntBitMask(const IsaVersion &Version) {
  return (1 << getSamplecntBitWidth(Version.Major)) - 1;
}

unsigned getBvhcntBitMask(const IsaVersion &Version) {
  return (1 << getBvhcntBitWidth(Version.Major)) - 1;
}

unsigned getExpcntBitMask(const IsaVersion &Version) {
  return (1 << getExpcntBitWidth(Version.Major)) - 1;
}

unsigned getLgkmcntBitMask(const IsaVersion &Version) {
  return (1 << getLgkmcntBitWidth(Version.Major)) - 1;
}

unsigned getDscntBitMask(const IsaVersion &Version) {
  return (1 << getDscntBitWidth(Version.Major)) - 1;
}

unsigned getKmcntBitMask(const IsaVersion &Version) {
  return (1 << getKmcntBitWidth(Version.Major)) - 1;
}

unsigned getXcntBitMask(const IsaVersion &Version) {
  return (1 << getXcntBitWidth(Version.Major, Version.Minor)) - 1;
}

unsigned getStorecntBitMask(const IsaVersion &Version) {
  return (1 << getStorecntBitWidth(Version.Major)) - 1;
}

unsigned getWaitcntBitMask(const IsaVersion &Version) {
  unsigned VmcntLo = getBitMask(getVmcntBitShiftLo(Version.Major),
                                getVmcntBitWidthLo(Version.Major));
  unsigned Expcnt = getBitMask(getExpcntBitShift(Version.Major),
                               getExpcntBitWidth(Version.Major));
  unsigned Lgkmcnt = getBitMask(getLgkmcntBitShift(Version.Major),
                                getLgkmcntBitWidth(Version.Major));
  unsigned VmcntHi = getBitMask(getVmcntBitShiftHi(Version.Major),
                                getVmcntBitWidthHi(Version.Major));
  return VmcntLo | Expcnt | Lgkmcnt | VmcntHi;
}

raw_ostream &operator<<(raw_ostream &OS, const AMDGPU::Waitcnt &Wait) {
  ListSeparator LS;
  if (Wait.LoadCnt != ~0u)
    OS << LS << "LoadCnt: " << Wait.LoadCnt;
  if (Wait.ExpCnt != ~0u)
    OS << LS << "ExpCnt: " << Wait.ExpCnt;
  if (Wait.DsCnt != ~0u)
    OS << LS << "DsCnt: " << Wait.DsCnt;
  if (Wait.StoreCnt != ~0u)
    OS << LS << "StoreCnt: " << Wait.StoreCnt;
  if (Wait.SampleCnt != ~0u)
    OS << LS << "SampleCnt: " << Wait.SampleCnt;
  if (Wait.BvhCnt != ~0u)
    OS << LS << "BvhCnt: " << Wait.BvhCnt;
  if (Wait.KmCnt != ~0u)
    OS << LS << "KmCnt: " << Wait.KmCnt;
  if (Wait.XCnt != ~0u)
    OS << LS << "XCnt: " << Wait.XCnt;
  if (LS.unused())
    OS << "none";
  OS << '\n';
  return OS;
}

unsigned decodeVmcnt(const IsaVersion &Version, unsigned Waitcnt) {
  unsigned VmcntLo = unpackBits(Waitcnt, getVmcntBitShiftLo(Version.Major),
                                getVmcntBitWidthLo(Version.Major));
  unsigned VmcntHi = unpackBits(Waitcnt, getVmcntBitShiftHi(Version.Major),
                                getVmcntBitWidthHi(Version.Major));
  return VmcntLo | VmcntHi << getVmcntBitWidthLo(Version.Major);
}

unsigned decodeExpcnt(const IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getExpcntBitShift(Version.Major),
                    getExpcntBitWidth(Version.Major));
}

unsigned decodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt) {
  return unpackBits(Waitcnt, getLgkmcntBitShift(Version.Major),
                    getLgkmcntBitWidth(Version.Major));
}

void decodeWaitcnt(const IsaVersion &Version, unsigned Waitcnt, unsigned &Vmcnt,
                   unsigned &Expcnt, unsigned &Lgkmcnt) {
  Vmcnt = decodeVmcnt(Version, Waitcnt);
  Expcnt = decodeExpcnt(Version, Waitcnt);
  Lgkmcnt = decodeLgkmcnt(Version, Waitcnt);
}

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded) {
  Waitcnt Decoded;
  Decoded.LoadCnt = decodeVmcnt(Version, Encoded);
  Decoded.ExpCnt = decodeExpcnt(Version, Encoded);
  Decoded.DsCnt = decodeLgkmcnt(Version, Encoded);
  return Decoded;
}

unsigned encodeVmcnt(const IsaVersion &Version, unsigned Waitcnt,
                     unsigned Vmcnt) {
  Waitcnt = packBits(Vmcnt, Waitcnt, getVmcntBitShiftLo(Version.Major),
                     getVmcntBitWidthLo(Version.Major));
  return packBits(Vmcnt >> getVmcntBitWidthLo(Version.Major), Waitcnt,
                  getVmcntBitShiftHi(Version.Major),
                  getVmcntBitWidthHi(Version.Major));
}

unsigned encodeExpcnt(const IsaVersion &Version, unsigned Waitcnt,
                      unsigned Expcnt) {
  return packBits(Expcnt, Waitcnt, getExpcntBitShift(Version.Major),
                  getExpcntBitWidth(Version.Major));
}

unsigned encodeLgkmcnt(const IsaVersion &Version, unsigned Waitcnt,
                       unsigned Lgkmcnt) {
  return packBits(Lgkmcnt, Waitcnt, getLgkmcntBitShift(Version.Major),
                  getLgkmcntBitWidth(Version.Major));
}

unsigned encodeWaitcnt(const IsaVersion &Version, unsigned Vmcnt,
                       unsigned Expcnt, unsigned Lgkmcnt) {
  unsigned Waitcnt = getWaitcntBitMask(Version);
  Waitcnt = encodeVmcnt(Version, Waitcnt, Vmcnt);
  Waitcnt = encodeExpcnt(Version, Waitcnt, Expcnt);
  Waitcnt = encodeLgkmcnt(Version, Waitcnt, Lgkmcnt);
  return Waitcnt;
}

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded) {
  return encodeWaitcnt(Version, Decoded.LoadCnt, Decoded.ExpCnt, Decoded.DsCnt);
}

static unsigned getCombinedCountBitMask(const IsaVersion &Version,
                                        bool IsStore) {
  unsigned Dscnt = getBitMask(getDscntBitShift(Version.Major),
                              getDscntBitWidth(Version.Major));
  if (IsStore) {
    unsigned Storecnt = getBitMask(getLoadcntStorecntBitShift(Version.Major),
                                   getStorecntBitWidth(Version.Major));
    return Dscnt | Storecnt;
  }
  unsigned Loadcnt = getBitMask(getLoadcntStorecntBitShift(Version.Major),
                                getLoadcntBitWidth(Version.Major));
  return Dscnt | Loadcnt;
}

Waitcnt decodeLoadcntDscnt(const IsaVersion &Version, unsigned LoadcntDscnt) {
  Waitcnt Decoded;
  Decoded.LoadCnt =
      unpackBits(LoadcntDscnt, getLoadcntStorecntBitShift(Version.Major),
                 getLoadcntBitWidth(Version.Major));
  Decoded.DsCnt = unpackBits(LoadcntDscnt, getDscntBitShift(Version.Major),
                             getDscntBitWidth(Version.Major));
  return Decoded;
}

Waitcnt decodeStorecntDscnt(const IsaVersion &Version, unsigned StorecntDscnt) {
  Waitcnt Decoded;
  Decoded.StoreCnt =
      unpackBits(StorecntDscnt, getLoadcntStorecntBitShift(Version.Major),
                 getStorecntBitWidth(Version.Major));
  Decoded.DsCnt = unpackBits(StorecntDscnt, getDscntBitShift(Version.Major),
                             getDscntBitWidth(Version.Major));
  return Decoded;
}

static unsigned encodeLoadcnt(const IsaVersion &Version, unsigned Waitcnt,
                              unsigned Loadcnt) {
  return packBits(Loadcnt, Waitcnt, getLoadcntStorecntBitShift(Version.Major),
                  getLoadcntBitWidth(Version.Major));
}

static unsigned encodeStorecnt(const IsaVersion &Version, unsigned Waitcnt,
                               unsigned Storecnt) {
  return packBits(Storecnt, Waitcnt, getLoadcntStorecntBitShift(Version.Major),
                  getStorecntBitWidth(Version.Major));
}

static unsigned encodeDscnt(const IsaVersion &Version, unsigned Waitcnt,
                            unsigned Dscnt) {
  return packBits(Dscnt, Waitcnt, getDscntBitShift(Version.Major),
                  getDscntBitWidth(Version.Major));
}

static unsigned encodeLoadcntDscnt(const IsaVersion &Version, unsigned Loadcnt,
                                   unsigned Dscnt) {
  unsigned Waitcnt = getCombinedCountBitMask(Version, false);
  Waitcnt = encodeLoadcnt(Version, Waitcnt, Loadcnt);
  Waitcnt = encodeDscnt(Version, Waitcnt, Dscnt);
  return Waitcnt;
}

unsigned encodeLoadcntDscnt(const IsaVersion &Version, const Waitcnt &Decoded) {
  return encodeLoadcntDscnt(Version, Decoded.LoadCnt, Decoded.DsCnt);
}

static unsigned encodeStorecntDscnt(const IsaVersion &Version,
                                    unsigned Storecnt, unsigned Dscnt) {
  unsigned Waitcnt = getCombinedCountBitMask(Version, true);
  Waitcnt = encodeStorecnt(Version, Waitcnt, Storecnt);
  Waitcnt = encodeDscnt(Version, Waitcnt, Dscnt);
  return Waitcnt;
}

unsigned encodeStorecntDscnt(const IsaVersion &Version,
                             const Waitcnt &Decoded) {
  return encodeStorecntDscnt(Version, Decoded.StoreCnt, Decoded.DsCnt);
}

} // end namespace AMDGPU
} // end namespace llvm
