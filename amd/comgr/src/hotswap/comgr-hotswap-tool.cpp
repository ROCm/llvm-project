//===- comgr-hotswap-tool.cpp - HSA_TOOLS_LIB B0->A0 rewrite tool ---------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// comgr hotswap tool (libamd_comgr_hotswap_tool.so), loaded via HSA_TOOLS_LIB.
// B0->A0 rewrite for gfx1250. Load with
// HSA_TOOLS_LIB=/path/libamd_comgr_hotswap_tool.so. Each gfx1250 A0 code object
// is rewritten via amd_comgr_hotswap_rewrite before reader creation; everything
// else passes through. HSA_HOTSWAP_TOOL_VERBOSE=1 for logging.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "inc/hsa.h"
#include "inc/hsa_api_trace.h"
#include "inc/hsa_ext_amd.h"
#include <amd_comgr.h>

#include "comgr-hotswap-tool-detect.h"

namespace COMGR::hotswap {

constexpr const char *Gfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

// Bare HSA_TOOLS_LIB callbacks carry no user-data, so all state lives here.
namespace {
struct HotswapTool {
  decltype(hsa_code_object_reader_create_from_memory) *RealReaderCreate =
      nullptr;
  decltype(hsa_iterate_agents) *IterateAgents = nullptr;
  decltype(hsa_agent_get_info) *AgentGetInfo = nullptr;
  decltype(hsa_isa_get_info_alt) *IsaGetInfoAlt = nullptr;
  CoreApiTable *Core = nullptr; // table we patch; restored in OnUnload

  bool Verbose = false;
  std::once_flag DetectOnce;
  bool DeviceIsA0 = false;

  // HSA references retained bytes for the module lifetime
  std::mutex RetainMutex;
  // Deque keeps element addresses stable
  std::deque<std::vector<uint8_t>> Retained;

  void detectDevice();
  void ensureDetected() {
    std::call_once(DetectOnce, [this] { detectDevice(); });
  }
};
} // namespace

// Function-local static singleton: one instance, thread-safe init, no globals.
// Detection is deferred (ensureDetected) because the HSA table is wired by
// OnLoad after construction and agents are only enumerable once HSA is up.
static HotswapTool &getTool() {
  static HotswapTool Tool;
  return Tool;
}

#define LOG(...)                                                               \
  do {                                                                         \
    if (getTool().Verbose) {                                                   \
      std::fprintf(stderr, "hotswap_tool: " __VA_ARGS__);                      \
      std::fprintf(stderr, "\n");                                              \
    }                                                                          \
  } while (0)

// First GPU agent: assume homogenous setup for Hotswap
static hsa_status_t findGpuAgent(hsa_agent_t Agent, void *Data) {
  hsa_device_type_t Type;
  if (getTool().AgentGetInfo(Agent, HSA_AGENT_INFO_DEVICE, &Type) ==
          HSA_STATUS_SUCCESS &&
      Type == HSA_DEVICE_TYPE_GPU) {
    *static_cast<hsa_agent_t *>(Data) = Agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

void HotswapTool::detectDevice() {
  if (!IterateAgents || !AgentGetInfo || !IsaGetInfoAlt) {
    LOG("detectDevice: HSA function pointers unavailable; rewrite disarmed");
    return;
  }
  hsa_agent_t Gpu = {0};
  IterateAgents(&findGpuAgent, &Gpu);
  if (Gpu.handle == 0) {
    LOG("detectDevice: no GPU agent found; rewrite disarmed");
    return;
  }

  // Size the ISA-name buffer from the queried length
  hsa_isa_t Isa = {0};
  std::string Gfx;
  if (AgentGetInfo(Gpu, HSA_AGENT_INFO_ISA, &Isa) == HSA_STATUS_SUCCESS) {
    uint32_t NameLen = 0;
    if (IsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME_LENGTH, &NameLen) ==
            HSA_STATUS_SUCCESS &&
        NameLen > 0) {
      std::vector<char> Name(NameLen + 1, '\0');
      if (IsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME, Name.data()) ==
          HSA_STATUS_SUCCESS) {
        Gfx = extractGfxTarget(Name.data());
      }
    }
  }

  // A failed revision query must not be mistaken for A0 (0); bail out disarmed.
  uint32_t Revision = 0;
  if (AgentGetInfo(
          Gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
          &Revision) != HSA_STATUS_SUCCESS) {
    LOG("detectDevice: ASIC revision query failed; rewrite disarmed");
    return;
  }

  DeviceIsA0 = gateAllowsHotswap(Gfx, Revision);
  LOG("device=%s asic_revision=%u -> %s", Gfx.empty() ? "?" : Gfx.c_str(),
      Revision, DeviceIsA0 ? "A0 (rewrite armed)" : "B0/native");
}

// Rewrite a gfx1250 object for A0; identity ISA, the A0 fix is in the rewrite.
static bool rewriteCodeObject(const void *Src, size_t Size,
                              std::vector<uint8_t> &Out) {
  amd_comgr_data_t Input = {0};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Input) !=
      AMD_COMGR_STATUS_SUCCESS) {
    LOG("comgr create_data FAILED");
    return false;
  }
  amd_comgr_data_t Output = {0};
  amd_comgr_status_t Status =
      amd_comgr_set_data(Input, Size, static_cast<const char *>(Src));
  if (Status == AMD_COMGR_STATUS_SUCCESS) {
    Status = amd_comgr_hotswap_rewrite(Input, Gfx1250Isa, Gfx1250Isa, &Output);
  } else {
    LOG("comgr set_data FAILED (status=%d)", static_cast<int>(Status));
  }
  amd_comgr_release_data(Input);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    LOG("comgr rewrite FAILED (status=%d)", static_cast<int>(Status));
    if (Output.handle) {
      amd_comgr_release_data(Output);
    }
    return false;
  }
  size_t OutSize = 0;
  Status = amd_comgr_get_data(Output, &OutSize, nullptr);
  if (Status == AMD_COMGR_STATUS_SUCCESS && OutSize > 0) {
    Out.resize(OutSize);
    Status = amd_comgr_get_data(Output, &OutSize,
                                reinterpret_cast<char *>(Out.data()));
  } else {
    LOG("comgr get_data returned no output (status=%d, size=%zu)",
        static_cast<int>(Status), OutSize);
    Status = AMD_COMGR_STATUS_ERROR;
  }
  amd_comgr_release_data(Output);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    LOG("comgr get_data FAILED (status=%d)", static_cast<int>(Status));
    return false;
  }
  LOG("comgr rewrite ok (%zu->%zu)", Size, OutSize);
  return true;
}

// Replaces hsa_code_object_reader_create_from_memory in the API table.
static hsa_status_t readerCreateWrapper(const void *CodeObject, size_t Size,
                                        hsa_code_object_reader_t *Reader) {
  HotswapTool &Tool = getTool();
  Tool.ensureDetected();

  // Forward invalid args and non-target objects to the real entry untouched.
  if (!CodeObject || !Size || !Reader || !Tool.DeviceIsA0 ||
      !isGfx1250CodeObject(CodeObject, Size)) {
    return Tool.RealReaderCreate(CodeObject, Size, Reader);
  }

  std::vector<uint8_t> Rewritten;
  if (!rewriteCodeObject(CodeObject, Size, Rewritten)) {
    // Warn unconditionally (not LOG): a failed rewrite forwards the original.
    std::fprintf(stderr, "hotswap_tool: rewrite failed; forwarding original\n");
    return Tool.RealReaderCreate(CodeObject, Size, Reader);
  }

  const uint8_t *Persisted;
  size_t PersistedSize;
  {
    const std::lock_guard<std::mutex> Lock(Tool.RetainMutex);
    Tool.Retained.emplace_back(std::move(Rewritten));
    Persisted = Tool.Retained.back().data();
    PersistedSize = Tool.Retained.back().size();
  }
  return Tool.RealReaderCreate(Persisted, PersistedSize, Reader);
}

} // namespace COMGR::hotswap

// OnLoad/OnUnload names and signatures are fixed by the HSA tool ABI.
// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" bool OnLoad(void *Table, uint64_t, uint64_t, const char *const *) {
  using namespace COMGR::hotswap;
  const HsaApiTable *Api = static_cast<const HsaApiTable *>(Table);
  if (!Api || !Api->core_) {
    return false;
  }
  CoreApiTable *Core = Api->core_;
  if (!Core->hsa_iterate_agents_fn || !Core->hsa_agent_get_info_fn ||
      !Core->hsa_isa_get_info_alt_fn ||
      !Core->hsa_code_object_reader_create_from_memory_fn) {
    return false;
  }
  if (Core->hsa_code_object_reader_create_from_memory_fn ==
      &readerCreateWrapper) {
    return true;
  }

  HotswapTool &Tool = getTool();
  if (const char *Verb = std::getenv("HSA_HOTSWAP_TOOL_VERBOSE")) {
    Tool.Verbose = Verb[0] && Verb[0] != '0';
  }
  Tool.IterateAgents = Core->hsa_iterate_agents_fn;
  Tool.AgentGetInfo = Core->hsa_agent_get_info_fn;
  Tool.IsaGetInfoAlt = Core->hsa_isa_get_info_alt_fn;
  Tool.RealReaderCreate = Core->hsa_code_object_reader_create_from_memory_fn;
  Tool.Core = Core;
  Core->hsa_code_object_reader_create_from_memory_fn = &readerCreateWrapper;
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" void OnUnload() {
  using namespace COMGR::hotswap;
  HotswapTool &Tool = getTool();
  // Restore the original entry so the table doesn't dangle after unload.
  if (Tool.Core && Tool.RealReaderCreate) {
    Tool.Core->hsa_code_object_reader_create_from_memory_fn =
        Tool.RealReaderCreate;
  }
}
