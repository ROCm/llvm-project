// Optional HSA_TOOLS_LIB hotswap tool: in-place B0->A0 rewrite for gfx1250.
//
// Enable by pointing HSA_TOOLS_LIB at this library:
//   HSA_TOOLS_LIB=/path/libamd_comgr_hotswap_tool.so ./my_app
//
// libhsa-runtime then hands each code object to the tool before dispatch. On a
// gfx1250 board treated as A0, each gfx1250 code object is rewritten in place
// via amd_comgr_hotswap_rewrite; everything else is passed through untouched.
//
// A0 detection reads HSA_AMD_AGENT_INFO_ASIC_REVISION from the HSA runtime
// (revision 0 == A0); the rewrite is gated to gfx1250 A0 only, and only when the
// revision was actually queried (a failed query is not treated as A0). No env
// var is required to enable it. Set HSA_HOTSWAP_TOOL_VERBOSE=1 for logging.
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "llvm/BinaryFormat/ELF.h"

#include "inc/hsa.h"
#include "inc/hsa_api_trace.h"
#include "inc/hsa_ext_amd.h"
#include <amd_comgr.h>

constexpr const char *Gfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

// All tool state lives in one object reached through getTool(). The
// HSA_TOOLS_LIB ABI installs our wrapper as a bare function pointer with no
// user-data parameter, so the wrapper cannot be handed a context and must reach
// this state on its own. A single function-local static keeps that contained
// (and free of static-initialization-order problems) instead of scattering
// globals.
namespace {
struct HotswapTool {
  // Loader entry we wrap, and the agent/ISA queries we need, from the table.
  decltype(hsa_code_object_reader_create_from_memory) *RealReaderCreate = nullptr;
  decltype(hsa_iterate_agents) *IterateAgents = nullptr;
  decltype(hsa_agent_get_info) *AgentGetInfo = nullptr;
  decltype(hsa_isa_get_info_alt) *IsaGetInfoAlt = nullptr;

  // Verbose logging, from HSA_HOTSWAP_TOOL_VERBOSE in OnLoad().
  bool Verbose = false;

  // Device facts, resolved once on the first code-object load.
  std::once_flag DetectOnce;
  bool DeviceIsA0 = false;

  // reader_create references the bytes for the module lifetime, so rewritten
  // buffers must outlive the call; retain them for the process lifetime.
  std::mutex RetainMutex;
  std::vector<std::vector<uint8_t>> Retained;

  void detectDevice();
  void ensureDetected() {
    std::call_once(DetectOnce, [this] { detectDevice(); });
  }
};
} // namespace

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

// hsa_iterate_agents callback: stop at the first GPU agent.
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

// Extracts the gfx target (e.g. "gfx1250") from a full HSA ISA name, dropping
// any feature suffix (":sramecc+", ":xnack-", ...) by stopping at the first
// non-alphanumeric character. Mirrors rocm-systems#7210's extract_gfx_target.
static std::string extractGfxTarget(const std::string &IsaName) {
  const size_t Start = IsaName.find("gfx");
  if (Start == std::string::npos) {
    return {};
  }
  size_t End = Start;
  while (End < IsaName.size() &&
         std::isalnum(static_cast<unsigned char>(IsaName[End]))) {
    ++End;
  }
  return IsaName.substr(Start, End - Start);
}

// HotSwap activation policy (mirrors rocm-systems#7210's gate_allows_hotswap):
// rewrite only on gfx1250 at ASIC revision A0 (0), and only when the revision
// was actually queried -- a failed query must not be treated as A0.
static bool gateAllowsHotswap(const std::string &Gfx, uint32_t Revision,
                              bool RevisionValid) {
  return RevisionValid && Gfx == "gfx1250" && Revision == 0;
}

void HotswapTool::detectDevice() {
  if (!IterateAgents || !AgentGetInfo || !IsaGetInfoAlt) {
    return;
  }
  hsa_agent_t Gpu = {0};
  IterateAgents(&findGpuAgent, &Gpu);
  if (Gpu.handle == 0) {
    return;
  }

  hsa_isa_t Isa = {0};
  char Name[128] = {0};
  std::string Gfx;
  if (AgentGetInfo(Gpu, HSA_AGENT_INFO_ISA, &Isa) == HSA_STATUS_SUCCESS &&
      IsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME, Name) == HSA_STATUS_SUCCESS) {
    Gfx = extractGfxTarget(Name);
  }

  // Trust the ASIC revision only when the query succeeds, so a failed query is
  // not mistaken for A0 (revision 0).
  uint32_t Revision = 0;
  const bool RevisionValid =
      AgentGetInfo(
          Gpu, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
          &Revision) == HSA_STATUS_SUCCESS;

  DeviceIsA0 = gateAllowsHotswap(Gfx, Revision, RevisionValid);
  LOG("device=%s asic_revision=%u (valid=%s) -> %s",
      Gfx.empty() ? "?" : Gfx.c_str(), Revision, RevisionValid ? "yes" : "no",
      DeviceIsA0 ? "A0 (rewrite armed)" : "B0/native");
}

// True for a 64-bit ELF whose AMDGPU mach selector is gfx1250. Uses the
// header-only enums/struct from llvm/BinaryFormat/ELF.h (compile-time constants,
// no linking, so no second in-process LLVM); only llvm::object's parsing classes
// are avoided.
static bool isGfx1250CodeObject(const void *Data, size_t Size) {
  if (Size < sizeof(llvm::ELF::Elf64_Ehdr)) {
    return false;
  }
  const llvm::ELF::Elf64_Ehdr *Header =
      static_cast<const llvm::ELF::Elf64_Ehdr *>(Data);
  return Header->checkMagic() &&
         Header->getFileClass() == llvm::ELF::ELFCLASS64 &&
         (Header->e_flags & llvm::ELF::EF_AMDGPU_MACH) ==
             llvm::ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250;
}

// Rewrite a gfx1250 code object for A0 via comgr (identity ISA, in place).
static bool rewriteCodeObject(const void *Src, size_t Size,
                              std::vector<uint8_t> &Out) {
  amd_comgr_data_t Input = {0};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Input) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  amd_comgr_data_t Output = {0};
  amd_comgr_status_t Status =
      amd_comgr_set_data(Input, Size, static_cast<const char *>(Src));
  if (Status == AMD_COMGR_STATUS_SUCCESS) {
    Status = amd_comgr_hotswap_rewrite(Input, Gfx1250Isa, Gfx1250Isa, &Output);
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
    Status = AMD_COMGR_STATUS_ERROR;
  }
  amd_comgr_release_data(Output);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  LOG("comgr rewrite ok (%zu->%zu)", Size, OutSize);
  return true;
}

// Installed into the HSA API table in place of
// hsa_code_object_reader_create_from_memory.
static hsa_status_t readerCreateWrapper(const void *CodeObject, size_t Size,
                                        hsa_code_object_reader_t *Reader) {
  HotswapTool &Tool = getTool();
  Tool.ensureDetected();

  if (!Tool.DeviceIsA0 || !isGfx1250CodeObject(CodeObject, Size)) {
    return Tool.RealReaderCreate(CodeObject, Size, Reader);
  }

  std::vector<uint8_t> Rewritten;
  if (!rewriteCodeObject(CodeObject, Size, Rewritten)) {
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

// HSA_TOOLS_LIB entry points. The OnLoad/OnUnload names and signature are fixed
// by the HSA tool ABI (libhsa-runtime looks them up by symbol and calls them
// with these exact types), so they cannot follow the camelBack naming rule or
// take a const Table parameter.
// NOLINTNEXTLINE(readability-identifier-naming,misc-const-correctness)
extern "C" bool OnLoad(void *Table, uint64_t, uint64_t, const char *const *) {
  const HsaApiTable *Api = static_cast<const HsaApiTable *>(Table);
  if (!Api || !Api->core_) {
    return false;
  }

  HotswapTool &Tool = getTool();
  if (const char *Verb = std::getenv("HSA_HOTSWAP_TOOL_VERBOSE")) {
    Tool.Verbose = Verb[0] && Verb[0] != '0';
  }

  Tool.IterateAgents = Api->core_->hsa_iterate_agents_fn;
  Tool.AgentGetInfo = Api->core_->hsa_agent_get_info_fn;
  Tool.IsaGetInfoAlt = Api->core_->hsa_isa_get_info_alt_fn;
  Tool.RealReaderCreate =
      Api->core_->hsa_code_object_reader_create_from_memory_fn;
  Api->core_->hsa_code_object_reader_create_from_memory_fn =
      &readerCreateWrapper;
  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
extern "C" void OnUnload() {}
