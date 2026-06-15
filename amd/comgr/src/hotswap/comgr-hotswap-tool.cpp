// Optional HSA_TOOLS_LIB hotswap tool: in-place B0->A0 rewrite for gfx1250.
//
// Enable by pointing HSA_TOOLS_LIB at this library:
//   HSA_TOOLS_LIB=/path/libamd_comgr_hotswap_tool.so ./my_app
//
// libhsa-runtime then hands each code object to the tool before dispatch. On a
// gfx1250 board treated as A0, each gfx1250 code object is rewritten in place
// via amd_comgr_hotswap_rewrite; everything else is passed through untouched.
//
// A0 selection is intentionally simple and is NOT robust stepping detection
// (separate work): rewrite when HSA_AMD_AGENT_INFO_ASIC_REVISION equals
// HSA_HOTSWAP_A0_REVISION (default 0), or when HSA_HOTSWAP_FORCE_STEPPING_REWRITE
// is set. Set HSA_HOTSWAP_TOOL_VERBOSE=1 for logging.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "inc/hsa.h"
#include "inc/hsa_ext_amd.h"
#include "inc/hsa_api_trace.h"
#include <amd_comgr.h>

namespace {

constexpr uint8_t kGfx1250Mach = 0x49;  // EF_AMDGPU_MACH for gfx1250
const char* const kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";

// Loader entry we wrap plus the agent/isa queries we need, from the API table.
decltype(hsa_code_object_reader_create_from_memory)* g_real_reader_create = nullptr;
decltype(hsa_iterate_agents)* g_iterate_agents = nullptr;
decltype(hsa_agent_get_info)* g_agent_get_info = nullptr;
decltype(hsa_isa_get_info_alt)* g_isa_get_info_alt = nullptr;

std::once_flag g_detect_once;
bool g_device_is_a0 = false;

uint32_t g_a0_revision = 0;
bool g_force_stepping_rewrite = false;
bool g_verbose = false;

// reader_create references the bytes for the module's lifetime, so rewritten
// buffers must outlive the call; retain them for the process lifetime.
std::mutex g_retain_mu;
std::vector<std::vector<uint8_t>> g_retained;

#define LOGF(...)                                                             \
  do {                                                                        \
    if (g_verbose) {                                                          \
      std::fprintf(stderr, "hotswap_tool: " __VA_ARGS__);                     \
      std::fprintf(stderr, "\n");                                             \
    }                                                                         \
  } while (0)

hsa_status_t FindGpuCb(hsa_agent_t agent, void* data) {
  hsa_device_type_t dt;
  if (g_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dt) == HSA_STATUS_SUCCESS &&
      dt == HSA_DEVICE_TYPE_GPU) {
    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

// Resolve the device ISA + stepping once (HSA is fully up by first load).
void DetectDevice() {
  if (!g_iterate_agents || !g_agent_get_info || !g_isa_get_info_alt) return;
  hsa_agent_t gpu = {0};
  g_iterate_agents(&FindGpuCb, &gpu);
  if (gpu.handle == 0) return;

  hsa_isa_t isa = {0};
  char name[128] = {0};
  std::string gfx;
  if (g_agent_get_info(gpu, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
      g_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name) == HSA_STATUS_SUCCESS) {
    std::string s(name);
    size_t g = s.find("gfx");
    if (g != std::string::npos) gfx = s.substr(g, s.find(':', g) - g);
  }

  uint32_t rev = 0;
  g_agent_get_info(gpu,
                   static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                   &rev);

  // gfx1250-only, so a different board reporting revision 0 is not armed.
  g_device_is_a0 = (gfx == "gfx1250") &&
                   (g_force_stepping_rewrite || rev == g_a0_revision);
  LOGF("device=%s asic_revision=%u -> %s", gfx.c_str(), rev,
       g_device_is_a0 ? "A0 (rewrite armed)" : "B0/native");
}

// True for a 64-bit ELF whose AMDGPU mach (e_flags low byte, offset 48) is
// gfx1250. We read fixed offsets rather than llvm::object: the tool links
// amd_comgr's symbol-hidden static LLVM and must not pull a second LLVM.
bool IsGfx1250Elf(const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  return n >= 64 && b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F' &&
         b[4] == 2 && b[48] == kGfx1250Mach;
}

// Rewrite a gfx1250 code object for A0 via comgr (identity ISA, in place).
bool Rewrite(const void* src, size_t size, std::vector<uint8_t>* out) {
  amd_comgr_data_t input = {0};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &input) !=
      AMD_COMGR_STATUS_SUCCESS)
    return false;
  amd_comgr_data_t output = {0};
  amd_comgr_status_t st =
      amd_comgr_set_data(input, size, static_cast<const char*>(src));
  if (st == AMD_COMGR_STATUS_SUCCESS)
    st = amd_comgr_hotswap_rewrite(input, kGfx1250Isa, kGfx1250Isa, &output);
  amd_comgr_release_data(input);
  if (st != AMD_COMGR_STATUS_SUCCESS) {
    LOGF("comgr rewrite FAILED (status=%d)", static_cast<int>(st));
    if (output.handle) amd_comgr_release_data(output);
    return false;
  }
  size_t osz = 0;
  st = amd_comgr_get_data(output, &osz, nullptr);
  if (st == AMD_COMGR_STATUS_SUCCESS && osz > 0) {
    out->resize(osz);
    st = amd_comgr_get_data(output, &osz, reinterpret_cast<char*>(out->data()));
  } else {
    st = AMD_COMGR_STATUS_ERROR;
  }
  amd_comgr_release_data(output);
  if (st != AMD_COMGR_STATUS_SUCCESS) return false;
  LOGF("comgr rewrite ok (%zu->%zu)", size, osz);
  return true;
}

hsa_status_t ReaderCreateWrapper(const void* code_object, size_t size,
                                 hsa_code_object_reader_t* reader) {
  std::call_once(g_detect_once, DetectDevice);

  if (!g_device_is_a0 || !IsGfx1250Elf(code_object, size))
    return g_real_reader_create(code_object, size, reader);

  std::vector<uint8_t> out;
  if (!Rewrite(code_object, size, &out)) {
    std::fprintf(stderr, "hotswap_tool: rewrite failed; forwarding original\n");
    return g_real_reader_create(code_object, size, reader);
  }

  const uint8_t* persist;
  size_t persist_size;
  {
    std::lock_guard<std::mutex> lk(g_retain_mu);
    g_retained.emplace_back(std::move(out));
    persist = g_retained.back().data();
    persist_size = g_retained.back().size();
  }
  return g_real_reader_create(persist, persist_size, reader);
}

}  // namespace

extern "C" bool OnLoad(void* table, uint64_t, uint64_t, const char* const*) {
  HsaApiTable* api = static_cast<HsaApiTable*>(table);
  if (!api || !api->core_) return false;

  if (const char* r = std::getenv("HSA_HOTSWAP_A0_REVISION"))
    g_a0_revision = static_cast<uint32_t>(std::strtoul(r, nullptr, 0));
  if (const char* f = std::getenv("HSA_HOTSWAP_FORCE_STEPPING_REWRITE"))
    g_force_stepping_rewrite = f[0] && f[0] != '0';
  if (const char* v = std::getenv("HSA_HOTSWAP_TOOL_VERBOSE"))
    g_verbose = v[0] && v[0] != '0';

  g_iterate_agents = api->core_->hsa_iterate_agents_fn;
  g_agent_get_info = api->core_->hsa_agent_get_info_fn;
  g_isa_get_info_alt = api->core_->hsa_isa_get_info_alt_fn;
  g_real_reader_create = api->core_->hsa_code_object_reader_create_from_memory_fn;
  api->core_->hsa_code_object_reader_create_from_memory_fn = &ReaderCreateWrapper;
  return true;
}

extern "C" void OnUnload() {}
