
#ifndef SANITIZER_H
#define SANITIZER_H

#include <cstdint>

// Architecture-agnostic structure to hold sanitizer data from one lane
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

// Architecture-agnostic handler for sanitizer reports
// Each plugin (AMDGPU, CUDA, etc.) should provide its own implementation
void HandleSanitizerReport(uint32_t NumLanes, const SanitizerData *LaneData,
                           uint64_t ActiveMask, int DeviceID);

#endif
