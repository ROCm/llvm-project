#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t Expected = 0xb0a00001u;

void check(hsa_status_t Status, const char *What) {
  if (Status == HSA_STATUS_SUCCESS)
    return;

  const char *Text = nullptr;
  hsa_status_string(Status, &Text);
  std::fprintf(stderr, "error: %s: %s\n", What, Text ? Text : "unknown");
  std::exit(1);
}

struct AgentState {
  hsa_agent_t Cpu{};
  hsa_agent_t Gpu{};
};

hsa_status_t findAgents(hsa_agent_t Agent, void *Data) {
  AgentState *State = static_cast<AgentState *>(Data);
  hsa_device_type_t Type;
  check(hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type),
        "hsa_agent_get_info(device)");
  if (Type == HSA_DEVICE_TYPE_CPU && State->Cpu.handle == 0)
    State->Cpu = Agent;
  if (Type == HSA_DEVICE_TYPE_GPU && State->Gpu.handle == 0)
    State->Gpu = Agent;
  return HSA_STATUS_SUCCESS;
}

struct PoolState {
  hsa_amd_memory_pool_t Kernarg{};
  hsa_amd_memory_pool_t Fine{};
};

hsa_status_t findPools(hsa_amd_memory_pool_t Pool, void *Data) {
  PoolState *State = static_cast<PoolState *>(Data);

  hsa_amd_segment_t Segment;
  check(hsa_amd_memory_pool_get_info(Pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                     &Segment),
        "hsa_amd_memory_pool_get_info(segment)");
  if (Segment != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;

  uint32_t Flags = 0;
  check(hsa_amd_memory_pool_get_info(
            Pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &Flags),
        "hsa_amd_memory_pool_get_info(global flags)");

  if ((Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) &&
      State->Kernarg.handle == 0)
    State->Kernarg = Pool;
  if ((Flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
      State->Fine.handle == 0)
    State->Fine = Pool;

  return HSA_STATUS_SUCCESS;
}

void queueError(hsa_status_t Status, hsa_queue_t *, void *) {
  const char *Text = nullptr;
  hsa_status_string(Status, &Text);
  std::fprintf(stderr, "queue error: %s\n", Text ? Text : "unknown");
}

} // namespace

int main(int Argc, char **Argv) {
  if (Argc < 2 || Argc > 3) {
    std::fprintf(stderr, "usage: %s <hsaco> [kernel]\n", Argv[0]);
    return 2;
  }

  const char *Hsaco = Argv[1];
  const char *Kernel = Argc == 3 ? Argv[2] : "b0_mask_rewrite_canary";

  check(hsa_init(), "hsa_init");

  AgentState Agents;
  check(hsa_iterate_agents(findAgents, &Agents), "hsa_iterate_agents");
  if (Agents.Gpu.handle == 0) {
    std::fprintf(stderr, "error: no GPU HSA agent found\n");
    return 1;
  }
  if (Agents.Cpu.handle == 0) {
    std::fprintf(stderr, "error: no CPU HSA agent found\n");
    return 1;
  }

  PoolState GpuPools;
  check(hsa_amd_agent_iterate_memory_pools(Agents.Gpu, findPools, &GpuPools),
        "hsa_amd_agent_iterate_memory_pools(gpu)");
  PoolState CpuPools;
  check(hsa_amd_agent_iterate_memory_pools(Agents.Cpu, findPools, &CpuPools),
        "hsa_amd_agent_iterate_memory_pools(cpu)");

  hsa_amd_memory_pool_t KernargPool =
      CpuPools.Kernarg.handle ? CpuPools.Kernarg : GpuPools.Kernarg;
  if (KernargPool.handle == 0) {
    std::fprintf(stderr, "error: no kernarg memory pool found\n");
    return 1;
  }
  hsa_amd_memory_pool_t OutputPool =
      CpuPools.Fine.handle ? CpuPools.Fine : GpuPools.Fine;
  if (OutputPool.handle == 0) {
    std::fprintf(stderr, "error: no fine-grained output memory pool found\n");
    return 1;
  }

  uint32_t *Output = nullptr;
  check(hsa_amd_memory_pool_allocate(OutputPool, sizeof(uint32_t), 0,
                                     reinterpret_cast<void **>(&Output)),
        "allocate output");
  *Output = 0;

  struct Kernarg {
    uint64_t OutputPtr;
  };
  Kernarg Args{reinterpret_cast<uint64_t>(Output)};

  void *KernargAddress = nullptr;
  check(hsa_amd_memory_pool_allocate(KernargPool, sizeof(Kernarg), 0,
                                     &KernargAddress),
        "allocate kernarg");
  std::memcpy(KernargAddress, &Args, sizeof(Args));

  hsa_agent_t AccessAgents[2] = {Agents.Cpu, Agents.Gpu};
  check(hsa_amd_agents_allow_access(2, AccessAgents, nullptr, Output),
        "allow output access");
  check(hsa_amd_agents_allow_access(2, AccessAgents, nullptr, KernargAddress),
        "allow kernarg access");

  int Fd = open(Hsaco, O_RDONLY);
  if (Fd < 0) {
    std::perror("open hsaco");
    return 1;
  }

  hsa_code_object_reader_t Reader;
  check(hsa_code_object_reader_create_from_file(Fd, &Reader),
        "hsa_code_object_reader_create_from_file");

  hsa_executable_t Executable;
  check(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &Executable),
        "hsa_executable_create_alt");
  check(hsa_executable_load_agent_code_object(Executable, Agents.Gpu, Reader,
                                              nullptr, nullptr),
        "hsa_executable_load_agent_code_object");
  check(hsa_executable_freeze(Executable, nullptr), "hsa_executable_freeze");

  hsa_executable_symbol_t Symbol;
  check(hsa_executable_get_symbol_by_name(Executable, Kernel, &Agents.Gpu,
                                          &Symbol),
        "hsa_executable_get_symbol_by_name");

  uint64_t KernelObject = 0;
  uint32_t PrivateSize = 0;
  uint32_t GroupSize = 0;
  check(hsa_executable_symbol_get_info(
            Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject),
        "get kernel object");
  check(hsa_executable_symbol_get_info(
            Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            &PrivateSize),
        "get private segment size");
  check(hsa_executable_symbol_get_info(
            Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            &GroupSize),
        "get group segment size");

  hsa_queue_t *Queue = nullptr;
  check(hsa_queue_create(Agents.Gpu, 128, HSA_QUEUE_TYPE_MULTI, queueError,
                         nullptr, UINT32_MAX, UINT32_MAX, &Queue),
        "hsa_queue_create");

  hsa_signal_t Signal;
  check(hsa_signal_create(1, 0, nullptr, &Signal), "hsa_signal_create");

  uint64_t Index = hsa_queue_add_write_index_relaxed(Queue, 1);
  uint32_t QueueMask = Queue->size - 1;
  hsa_kernel_dispatch_packet_t *Packet =
      &reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
          Queue->base_address)[Index & QueueMask];
  std::memset(Packet, 0, sizeof(*Packet));

  Packet->setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet->workgroup_size_x = 1;
  Packet->workgroup_size_y = 1;
  Packet->workgroup_size_z = 1;
  Packet->grid_size_x = 1;
  Packet->grid_size_y = 1;
  Packet->grid_size_z = 1;
  Packet->private_segment_size = PrivateSize;
  Packet->group_segment_size = GroupSize;
  Packet->kernel_object = KernelObject;
  Packet->kernarg_address = KernargAddress;
  Packet->completion_signal = Signal;

  uint32_t Header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(&Packet->header), Header,
                   __ATOMIC_RELEASE);
  hsa_signal_store_release(Queue->doorbell_signal, Index);

  hsa_signal_value_t Done = hsa_signal_wait_acquire(
      Signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (Done != 0) {
    std::fprintf(stderr, "error: dispatch did not complete, signal=%ld\n",
                 static_cast<long>(Done));
    return 1;
  }

  if (*Output != Expected) {
    std::fprintf(stderr, "error: output 0x%08x != 0x%08x\n", *Output, Expected);
    return 1;
  }

  std::printf("PASS: %s wrote 0x%08x\n", Kernel, *Output);

  check(hsa_signal_destroy(Signal), "hsa_signal_destroy");
  check(hsa_queue_destroy(Queue), "hsa_queue_destroy");
  check(hsa_executable_destroy(Executable), "hsa_executable_destroy");
  check(hsa_code_object_reader_destroy(Reader),
        "hsa_code_object_reader_destroy");
  close(Fd);
  check(hsa_amd_memory_pool_free(KernargAddress), "free kernarg");
  check(hsa_amd_memory_pool_free(Output), "free output");
  check(hsa_shut_down(), "hsa_shut_down");
  return 0;
}
