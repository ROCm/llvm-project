
#ifndef SANITIZER_H
#define SANITIZER_H

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <vector>

#if defined(__has_include)
#if __has_include("hsa.h")
#include "hsa.h"
#include "hsa_ven_amd_loader.h"
#elif __has_include("hsa/hsa.h")
#include "hsa/hsa.h"
#include "hsa/hsa_ven_amd_loader.h"
#endif
#else
#include "hsa/hsa.h"
#include "hsa/hsa_vem_amd_loader.h"
#endif

// Structure to hold sanitizer data from one lane
struct SanitizerData {
  uint64_t addr;
  uint64_t pc;
  uint64_t wgidx;
  uint64_t wgidy;
  uint64_t wgidz;
  uint64_t wave_id;
  uint64_t is_read;
  uint64_t access_size;
};

class UriLocator {
public:
  struct UriInfo {
    std::string uriPath;
    int64_t loadAddressDiff;
  };

  struct UriRange {
    uint64_t startAddr_, endAddr_;
    int64_t elfDelta_;
    std::string Uri_;
  };

  bool init_ = false;
  std::vector<UriRange> rangeTab_;
  hsa_ven_amd_loader_1_03_pfn_t fn_table_;

  hsa_status_t createUriRangeTable();

  ~UriLocator() {}

  UriInfo lookUpUri(uint64_t device_pc);
  std::pair<uint64_t, uint64_t> decodeUriAndGetFd(UriInfo &uri_path,
                                                  int *uri_fd);
};

void HandleSanitizerReport(uint32_t NumLanes, const SanitizerData *LaneData,
                           uint64_t ActiveMask, int DeviceID);
#endif
