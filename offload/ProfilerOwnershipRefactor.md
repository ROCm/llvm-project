# Profiler Ownership Refactor — Implementation Plan

Status: **draft for review**, not yet implemented. No code has been changed.
Scope: `offload/` runtime (libomptarget + plugins-nextgen), AMDGPU plugin
receives the deepest treatment because of its asynchronous HSA completion
machinery. CUDA / host / level_zero are touched only at the common-API
level. Rewriting libomptarget in terms of `liboffload` is explicitly **not**
part of this work; `liboffload` is treated as a second, independent
consumer of the plugin layer that needs its own (possibly no-op) profiler
source.

All file:line references are against branch `ompt-profiler-rework` as of
2026-08-12 and will drift as the tree changes; treat them as pointers, not
promises.

---

## 1. Goals

1. Remove link-time selection of the profiler (the weak/strong
   `getProfilerToAttach()` pair). A single concrete profiler type is chosen
   by whoever constructs it, not by the linker.
2. Make the profiler an explicit, ordinary object in the API instead of
   something plugins reach for implicitly via `Plugin.getProfiler()`. This
   is what unblocks other language runtimes built on `plugins-nextgen`
   (e.g. `liboffload`, or a future non-OpenMP consumer) from supplying
   their own profiler instance instead of inheriting whatever
   `libomptarget` happened to link in.
3. Move profiler **ownership** out of `GenericPluginTy` and into
   `libomptarget` (concretely: `PluginManager`). When OMPT is compiled in,
   that instance is `OmptProfilerTy`; otherwise it is a no-op
   `GenericProfilerTy`. There is exactly one profiler instance for the
   whole process, not one per plugin/backend.
4. Preserve today's behavior and performance for the OMPT-enabled and
   OMPT-disabled builds. This is a structural refactor, not a feature
   change.
5. Make teardown provably safe through explicit lifecycle postconditions:
   successful plugin deinitialization must complete all device work and execute
   every deferred profiling slot action; profiler deinitialization then verifies
   that no measurement remains outstanding, drains OMPT trace-buffer delivery,
   and only afterward releases tracing resources.

## 2. Non-goals

- Rewriting `libomptarget` in terms of `liboffload`. `liboffload` is
  updated only enough to keep building and to get a reasonable (likely
  no-op) profiler of its own; it does not get OMPT wiring in this patch
  series.
- Removing the AMDGPU `#ifdef OMPT_SUPPORT` compile-time gates in
  `rtl.cpp`. There are ~9 of them; cleaning them up is valuable but
  orthogonal and deferred to a follow-up (see §10).
- Changing the `TracerInterfaceRAII`/`InterfaceRAII` callback mechanism
  itself, or the OMPT trace-record wire format.
- Introducing a second OMPT "tracing session" abstraction. For this series,
  `OmptProfilerTy` itself is the cohesive owner of OMPT measurement tracking
  and trace-buffer management.
- Supporting changes between tracing-enabled and tracing-disabled state while
  device operations are in flight. This series starts with the explicit
  assumption that tracing state is fixed for the runtime's execution phase and
  encodes that assumption with assertions.

## 3. Current architecture

```
GenericPluginTy                              (one per backend, e.g. amdgpu)
  └─ std::unique_ptr<GenericProfilerTy> Profiler   ← constructed via
                                                      getProfilerToAttach()
                                                      at GenericPluginTy
                                                      construction time
GenericDeviceTy (many per plugin)
  └─ GenericPluginTy &Plugin                  ← devices reach the profiler
                                                 via Plugin.getProfiler()
```

`getProfilerToAttach()` is a weak symbol defined in
`offload/plugins-nextgen/common/src/GenericProfiler.cpp:20` (returns a
no-op `GenericProfilerTy`) and a strong override in
`offload/libomptarget/PluginManager.cpp:654-664` that returns an
`OmptProfilerTy` when `OMPT_SUPPORT` is defined. Because it's link-time
symbol resolution, **whichever definition ends up in the final link wins
for every plugin**, and every plugin gets its own separate profiler
instance — there is no sharing, no explicit hand-off, and no way for a
non-`libomptarget` consumer (like `liboffload`) to supply a different one
without also linking a conflicting strong symbol.

Every device operation that needs profiling data reaches it by walking
`<device>.Plugin.getProfiler()` — see the full call-graph in §5.

## 4. Target architecture

```
PluginManager (global "PM", libomptarget)
  └─ std::unique_ptr<GenericProfilerTy> Profiler   ← constructed once in
                                                       PluginManager::init(),
                                                       OmptProfilerTy iff
                                                       OMPT_SUPPORT, else a
                                                       no-op GenericProfilerTy
       └─ OmptProfilerTy only:
            ├─ OmptTracingBufferMgr TraceRecordManager
            └─ ProfilerData map                    ← accepted measurements whose
                                                       backend completion action
                                                       has not yet run
GenericPluginTy         (no longer owns or constructs a profiler)
GenericDeviceTy
  ...launchKernel(..., GenericProfilerTy &Profiler)
  ...dataAlloc(..., GenericProfilerTy &Profiler)
  ...                                          ← profiler flows in as an
                                                  explicit parameter from
                                                  whoever has it (ultimately
                                                  libomptarget's PM, or
                                                  liboffload's own instance)
```

Key decisions this section fixes (see §9 for the reasoning):

- **Reference, not pointer, for synchronous call chains.** Every
  synchronous plugin-common API (`init`, `deinit`, `loadBinary`,
  `dataAlloc`, `dataDelete`, `launch`/`launchKernel`, and the C-API
  boundary methods on `GenericPluginTy`) takes `GenericProfilerTy
  &Profiler`. There is always a live profiler to pass — worst case the
  no-op — so a reference is appropriate and self-documents "this can never
  be null."
- **Pointer for anything that outlives the call that created it.**
  Anything captured into `ProfilingInfoTy`, an `AsyncInfoTy::ProfilerData`
  payload, or a stream-slot completion action keeps using a raw
  `GenericProfilerTy *`, exactly as `void *ProfilerSpecificData` and
  `TracerInterfaceRAII`'s `GenericProfilerTy *Prof` already do today. The
  pointer is safe under the ownership/lifetime rule in §7, not because it
  is nullable — it should never actually be null in practice, but we
  don't want to encode "this is guaranteed non-dangling forever" as a C++
  reference across an async boundary we don't fully control.
- **`PluginManager` is the sole owner of libomptarget's profiler.** Not
  `PluginContextTy` (that's a `liboffload`-only per-context object; using it
  would only benefit `liboffload`, not the actual ask), and not a static/global
  inside `GenericProfiler.cpp` (that resurrects the link-time-selection
  problem in a different shape). `PluginManager` knows only the generic
  profiler lifecycle and invokes `Profiler->deinit()`; it does not inspect or
  coordinate OMPT implementation details.
- **`OmptProfilerTy` owns `OmptTracingBufferMgr`.** The buffer manager and the
  `ProfilerData` map describe two stages of the same OMPT tracing pipeline, so
  their lifetime belongs under one OMPT-specific owner. OMPT tracing code uses
  the typed profiler/buffer-manager relationship established when the concrete
  profiler is constructed; this does not require casts in generic plugin or
  `PluginManager` code.
- **No-op profiler always exists, even without OMPT.** This means
  `PM->getProfiler()` is never null and callers never need a null check
  before calling into it — matching today's `GenericProfilerTy`'s
  all-no-op-default design.

## 5. Complete signature-change inventory

This is the union of both investigation passes. "STOP" marks the point in
each chain where `libomptarget` already has (or trivially can get) `PM`
in scope, i.e. where the parameter's *source* changes but the chain of
callers threading it through ends.

### 5.1 `plugins-nextgen/common` — needs a new `GenericProfilerTy &Profiler`
(or, for AMDGPU async captures, `GenericProfilerTy *`) parameter

| Function | Definition | Current profiler access | Callers (→ next hop) |
|---|---|---|---|
| `GenericDeviceTy::init(GenericPluginTy&)` | `PluginInterface.cpp:651` | `Plugin.getProfiler()` (652,657,660) | `GenericPluginTy::initDevice` (1550) → `init_device` (1686, C-API) → **STOP** `DeviceTy::init()` `device.cpp:92`; also `liboffload/OffloadImpl.cpp:148` |
| `GenericDeviceTy::deinit(GenericPluginTy&)` | `PluginInterface.cpp:731` | `Plugin.getProfiler()` (770) | `GenericPluginTy::deinitDevice` (1559) → `GenericPluginTy::deinit()` (1515, C-API) → **STOP** `PluginManager.cpp:84`; also `liboffload/OffloadImpl.cpp:134` |
| `GenericDeviceTy::loadBinary(GenericPluginTy&, StringRef)` | `PluginInterface.cpp:776` | `Plugin.getProfiler()` (819) | `GenericPluginTy::load_binary` (1784, C-API) → **STOP** `device.cpp:228`; also `liboffload/OffloadImpl.cpp:1148` |
| `GenericDeviceTy::dataAlloc(...)` | `PluginInterface.cpp:1100` | `Plugin.getProfiler()` (1106, **unconditional deref, no null check today**) | `GenericKernelTy::getKernelLaunchEnvironment` (184,205) and `prepareBlockMemory` (303) — both part of the launch chain (§5.1 launch row); `GenericPluginTy::data_alloc` (1808, C-API) → **STOP** `device.cpp:292`; also `liboffload/OffloadImpl.cpp:693` |
| `GenericDeviceTy::dataDelete(...)` | `PluginInterface.cpp:1174` | `Plugin.getProfiler()` (1177, unconditional) | internal free-after-sync loop (`PluginInterface.cpp:1068`); `GenericPluginTy::data_delete` (1829, C-API) → **STOP** `device.cpp:307`; also `liboffload/OffloadImpl.cpp:725,790` |
| `GenericKernelTy::launch(...)` | `PluginInterface.cpp:316` | `GenericDevice.Plugin.getProfiler()` (437-438, `handlePreKernelLaunch`) + transitively via `dataAlloc` | `GenericDeviceTy::launchKernel` (1316) → `GenericPluginTy::launch_kernel` (2041, C-API) → **STOP** `device.cpp:430`, `omptarget.cpp:2383`, `omptarget.cpp:2527`; also `liboffload/OffloadImpl.cpp:1274` |

All the "C-API boundary" methods above (`GenericPluginTy::initDevice`,
`deinitDevice`, `deinit`, `init_device`, `load_binary`, `data_alloc`,
`data_delete`, `launch_kernel`) are **not** the `extern "C"`
`createPlugin_*` ABI boundary — they are ordinary virtual/non-virtual C++
methods on `GenericPluginTy` that happen to mirror the historical
`__tgt_rtl_*` C API shape. They are statically linked into `omptarget`
(see §6), so changing their signatures is a source change requiring a
rebuild, not an ABI break between independently shipped binaries.

### 5.2 AMDGPU plugin (`plugins-nextgen/amdgpu/src/rtl.cpp`)

| Function | Definition | Notes |
|---|---|---|
| `AMDGPUQueueTy::init(...)` | `:1499` | `Device.Plugin.getProfiler()->isProfilingEnabled()` at `:1507`, unconditional deref |
| `AMDGPUQueueManagerTy::init` / `::assignNextQueue` | `:3009` / `:3063` | callers of the above, already have `Device` in scope |
| `AMDGPUDeviceTy::dataSubmitImpl/dataRetrieveImpl/dataExchangeImpl` | `:4115/:4206/:4325` | `Plugin.getProfiler()->isProfilingEnabled()` (4175/4267/4352); each builds a `ProfilingInfoTy{&Plugin, ...}` — this is the struct that needs to carry `GenericProfilerTy*` instead of `GenericPluginTy*` |
| `AMDGPUDeviceTy::deriveHostToDeviceClockOffset` | `:5529` | `Plugin.getProfiler()->setTimeConversionFactors(...)` (5535); called from `initImpl(GenericPluginTy&)` at `:3632`, which already receives the plugin — trivial to also receive/derive the profiler via item 5.1's `init` chain |
| `AMDGPUStreamTy::pushKernelLaunch` | `:2284` | calls `schedProfilerKernelTiming` (2315) |
| `AMDGPUStreamTy::pushPinnedMemoryCopyAsync` | `:2389` | calls `schedProfilerDataTransferTiming` (2407) |
| `AMDGPUStreamTy::pushMemoryCopyD2HAsync` | `:2432` | `schedProfilerDataTransferTiming` (2463) |
| `AMDGPUStreamTy::pushMemoryCopyH2DAsync` | `:2519` | `schedProfilerDataTransferTiming` (2587) |
| `AMDGPUStreamTy::pushMemoryCopyD2DAsync` | `:2608` | `schedProfilerDataTransferTiming` (2626) |
| `schedProfilerKernelTiming` / `schedProfilerDataTransferTiming` | `:1881` / `:1893` | **This is where `&(Device->Plugin)` becomes the captured `GenericProfilerTy*`.** Currently builds `ProfilingInfoTy{&(Device->Plugin), Agent, OutputSignal, TicksToTime, ProfilerSpecificData}` |
| `timeKernelInNsAsync` / `timeDataTransferInNsAsync` | `:2252` / `:158` | dereference `ProfilerInfo->Plugin->getProfiler()` — becomes `ProfilerInfo->Profiler->handle...()` directly |
| `ProfilingInfoTy` | `:127-142` | change member `GenericPluginTy *Plugin;` → `GenericProfilerTy *Profiler;`. Also read (assert-only) in `getProfilingInfo` (:6615), `getKernelStartAndEndTime` (:6627) |

### 5.3 CUDA / host / level_zero — ripple only

Neither CUDA nor host calls `getProfiler()` anywhere; the only trace is a
redundant `extern getProfilerToAttach();` forward declaration
(`cuda/src/rtl.cpp:47`, `host/src/rtl.cpp:51`), which is simply deleted.
Level_zero has no profiler references at all. All three inherit the base
`GenericDeviceTy`/`GenericKernelTy` methods, so once those signatures
change, CUDA/host/L0 compile against the new signatures automatically —
**unless** we also change the virtual signature of
`dataSubmitImpl`/`dataRetrieveImpl`/`dataExchangeImpl` (AMDGPU overrides
these; CUDA/host/L0 do too, without touching the profiler:
`cuda/src/rtl.cpp:820,834,863,1821`; `host/src/rtl.cpp:280,287,301`;
`L0Device.cpp:338,358,395`; `L0Queue.cpp:254,294`). **Decision: do not
change the virtual `*Impl` signature.** Keep the profiler flowing through
`ProfilingInfoTy` (AMDGPU-internal) and `AsyncInfo->ProfilerData`
(cross-plugin), exactly as today's `void *ProfilerSpecificData` pattern
already does, so CUDA/host/L0 need zero changes here.

### 5.4 libomptarget — the "STOP" points, now sources of the profiler

Four sites already pass a profiler explicitly to `TracerInterfaceRAII`
and simply switch their source from `RTL->getProfiler()` /
`Device.RTL->getProfiler()` to `PM->getProfiler()`:

- `device.cpp:329` in `DeviceTy::submitData(...)`
- `device.cpp:359` in `DeviceTy::retrieveData(...)`
- `device.cpp:388` in `DeviceTy::dataExchange(...)`
- `omptarget.cpp:2380` in `target(...)`

The remaining STOP points (`DeviceTy::init/allocData/deleteData/
launchKernel`, `target`, `target_replay`) currently call straight through
to the plugin without touching the profiler at all; they gain a
`PM->getProfiler()` argument at the call site where they invoke the
now-changed plugin-common methods from §5.1.

### 5.5 `liboffload` — a separate consumer, needs its own profiler source

`liboffload/src/OffloadImpl.cpp` calls straight into `GenericPluginTy`/
`GenericDeviceTy`/`GenericKernelTy` methods without ever going through
`libomptarget` or `PM`:
`initDevice` (148), `deinit` (134), `loadBinary` (1148), `dataAlloc`
(693), `dataDelete` (725, 790), `GenericKernelTy::launch` directly via
`olLaunchKernel_impl` (1274).

Since `liboffload` doesn't link `libomptarget`, it cannot reach `PM`.
**Decision:** give `liboffload` its own no-op `GenericProfilerTy`
instance (e.g. a single static/`ol_context`-scoped no-op profiler,
constructed the same way the plugin-common weak fallback works today,
minus the link-time-selection problem — it's a compile-time `new
GenericProfilerTy()` in `liboffload`'s own source, not a weak symbol
plugins guess at). This is a small, mechanical change and is explicitly
in scope (it must compile and work, just without OMPT wiring — see
non-goals). If `liboffload` ever wants OMPT support, it constructs an
`OmptProfilerTy` the same way `PluginManager` does; that's future work.

## 6. ABI / build analysis (unchanged conclusion, reconfirmed)

Plugins in `plugins-nextgen/*` are `STATIC` libraries
(`plugins-nextgen/CMakeLists.txt:6`), privately linked into `omptarget`
(`libomptarget/CMakeLists.txt`), and never installed standalone. The only
extern-C boundary is `createPlugin_<name>()`
(`PluginManager.cpp` / `rtl.cpp:6823-6825` for AMDGPU), whose signature
(`GenericPluginTy *createPlugin_amdgpu()`) is **not** touched by this
refactor. Every signature change in §5 is therefore a coordinated
in-tree source change requiring a full rebuild of `omptarget` and its
statically-linked plugins — not a compatibility break between
independently versioned/shipped binaries. `liboffload` links the plugins
independently (its own `PluginContextTy`/`GenericPluginTy` instances,
per `liboffload/src/OffloadImpl.cpp`), which is exactly why §5.5 exists:
it is a second static-link consumer of the same in-tree interface, not an
ABI concern.

## 7. Profiler-data ownership and async-lifetime rules

This is the crux of the "is it safe" question and is grounded directly
in the teardown investigation (§8). The design uses two distinct completion
stages and gives each one an explicit postcondition.

### 7.1 Measurement completion versus trace-record delivery

`OmptProfilerTy::ProfilerData` and `OmptTracingBufferMgr` do not represent the
same kind of outstanding work:

- An entry in `ProfilerData` is an accepted measurement whose backend profiling
  completion action has not yet executed. The entry owns `OmptEventInfoTy`,
  which points at its trace record in buffer-manager storage.
- A ready trace-buffer record represents a completed measurement that may not
  yet have been delivered to the OMPT tool by a buffer helper thread.

Accordingly, map emptiness means that all backend measurement callbacks have
finished; it does **not** mean that all records have been delivered. Final
profiler teardown first asserts the former and then waits for the latter.

### 7.2 Lifetime rules

1. **Runtime deinitialization closes admission before plugin teardown.** No new
   device operation or profiler-data entry may be created after runtime
   deinitialization begins. This series assumes tracing is either enabled or
   disabled for the runtime's execution phase; changing that state while
   operations are in flight is unsupported and asserted against.
2. **Successful plugin deinitialization completes device work and all associated
   host-side completion actions.** "The kernel/copy finished" is not sufficient:
   AMDGPU stream-slot `performAction()` callbacks, including
   `timeKernelInNsAsync` and `timeDataTransferInNsAsync`, must also have run.
   Timestamp-query failure may produce the existing zero/junk timing values, but
   it still completes the OMPT record and retires its profiler-data entry.
3. **`ProfilerData` remains owned by `OmptProfilerTy`.** Creation inserts one
   entry; every successful or failed terminal path after insertion must retire
   exactly that entry. Immediately before `Profiler->deinit()`, the map must be
   empty. `OmptProfilerTy::deinit()` checks this as a plugin-teardown
   postcondition; it does not wait for backend/HSA progress.
4. **`OmptProfilerTy` owns `OmptTracingBufferMgr`.** Once the measurement map is
   empty and runtime admission is closed, no producer can create or complete
   another asynchronous trace record. `OmptProfilerTy::deinit()` then flushes
   all ready records, waits for buffer-completion callbacks, stops and joins the
   helper threads, and releases the buffer manager.
5. **Captured profiler pointers are stable through plugin teardown.** AMDGPU's
   `ProfilingInfoTy`, `AsyncInfo->ProfilerData`, and other async state refer to
   the `PluginManager`-owned profiler until all slot actions have executed.
   The profiler is destroyed only after plugin deinitialization and successful
   profiler deinitialization.
6. **A plugin-deinit failure cannot be followed by blind profiler destruction.**
   If a plugin cannot establish the map-empty postcondition, shutdown must use
   an explicit safe failure policy rather than log-and-continue into destruction
   of objects still referenced by completion state.

## 8. Teardown/shutdown findings (from investigation) and required fixes

This section documents what a dedicated read-only investigation found in
the current tree, because the lifecycle postconditions in §7 depend on it.

### 8.1 Current shutdown order (as-is, before this refactor)

`__tgt_rtl_deinit()` → `deinitRuntime()` (`OffloadRTL.cpp:64-83`):
1. Spin-wait for `RTLOngoingSyncs == 0` (`:72-76`) — **this only guards
   explicit interop syncs (`__tgt_target_sync`,
   `libomptarget/OpenMP/API.cpp:842-850`), not general async offload
   completion.** It does not wait for outstanding kernel/copy signals.
2. `PM->deinit()` (`:77`)
3. `delete PM` (`:78`)

Separately, `PluginManager::unregisterLib()` (called earlier, before
`deinitRuntime`, at library-unload time) calls
`OMPT_IF_TRACING_ENABLED(PM->getTraceRecordManager()->
flushAndShutdownHelperThreads())` (`PluginManager.cpp:396-399`) — this is
"best effort": `OmptTracingBufferMgr::flushAllBuffers` only flushes
records that worker threads have already marked ready
(`OmptTracingBuffer.cpp:590-661`); it has **no visibility into and does
not wait for in-flight HSA async completion actions**. The manager's own
comments (`:239-244, 375-378`) acknowledge records can be dropped and
buffer-completion callbacks can fire after tracing is disabled.

`PluginManager::deinit()` itself (`PluginManager.cpp:65-92`):
1. **Deletes `TraceRecordManager` first** (`:73-78`), *before* touching
   any plugin.
2. Then for each plugin: `Plugin->deinit()` (`:84`) followed by
   **`Plugin.release()`** (`:88`) — the plugin object is **never
   `delete`d**; it, and its `Profiler` unique_ptr, are leaked for the
   life of the process (only reclaimed by process exit).

Neither `GenericPluginTy::deinit()` nor `GenericDeviceTy::deinit()`
(`PluginInterface.cpp:1509-1537`, `:731-774`) call `synchronize()` on any
device. AMDGPU's `AMDGPUDeviceTy::deinitImpl` (`rtl.cpp:3784-3801`) and
`AMDGPUStreamManagerTy::deinit` (`rtl.cpp:3023-3031`) likewise never
synchronize in-use streams; the resource-pool `deinit`
(`PluginInterface.h:2266-2279`) only warns if resources are still
checked out and destroys the idle remainder. Nothing deregisters a
pending `hsa_amd_signal_async_handler` registration
(`rtl.cpp:2506-2508, 2552-2554`), which holds a **raw pointer into a
stream's `Slots` deque** — if that stream/device were destroyed with the
handler still pending, the handler thread would dereference freed memory.

### 8.2 Why this is safe *today* (and why that stops being true)

Two independent accidents currently prevent a profiler use-after-free:

1. **The profiler is never destroyed** — `Plugin.release()` leaks it, so
   there is nothing to free.
2. **No profiler-touching code runs on the HSA async-handler thread.**
   `timeKernelInNsAsync`/`timeDataTransferInNsAsync` are dispatched from
   `performAction()` (`rtl.cpp:1906`), which is invoked from `complete()`/
   `completeUntil()` on the **synchronizing application thread**
   (`rtl.cpp:2052,2077`), not from `asyncActionCallback`
   (`rtl.cpp:2121`, the actual HSA-handler-thread entry point, used only
   for the two-step host-to-host memcpy and RPC notification, neither of
   which touches the profiler).

**This refactor removes accident #1 by design** — a `PluginManager`-owned
profiler must have a real, finite lifetime for ownership to mean anything.
The corrected plan does not make accident #2 a lifetime guarantee. Instead,
each backend's successful `deinit()` must establish that all device work and
all associated host-side slot actions have completed before profiler teardown
begins (§8.3). This contract remains valid if a backend later moves profiler
work onto an async-handler thread, provided that thread is also joined/drained
before backend deinit reports success.

### 8.3 Required fix: encode plugin-drain and profiler-shutdown postconditions

The earlier version of this plan proposed that `PluginManager` manufacture a
`GenericDeviceTy::synchronize` call for every device. That is not the right
abstraction: `GenericDeviceTy::synchronize` requires a particular
`__tgt_async_info`, while the backend owns the queues and stream-slot actions
that must be drained. The corrected division of responsibility is:

- Runtime deinitialization prevents submission of new work.
- Each initialized plugin's `deinit()` drains its backend-owned queues and
  executes all pending completion/slot actions before it reports success.
  AMDGPU must therefore establish that no consumed stream slot, pending action,
  or HSA handler can still reference profiler/device state.
- `PluginManager::deinit()` invokes plugin deinitialization while the shared
  profiler remains alive. It does not inspect OMPT state.
- After all plugins report successful deinitialization, `PluginManager` calls
  the generic `Profiler->deinit()` hook.
- `OmptProfilerTy::deinit()` asserts, under `ProfilerDataMutex`, that
  `ProfilerData` is empty. A nonempty map identifies a plugin drain or
  measurement-retirement bug; profiler deinit must not block hoping to drive
  backend progress.
- `OmptProfilerTy::deinit()` then performs the final trace-buffer flush, waits
  for all buffer-completion callbacks, shuts down and joins helper threads, and
  releases its owned `OmptTracingBufferMgr`.

This fixes the current ordering error in `PluginManager::deinit()`, which deletes
`TraceRecordManager` before plugin deinitialization even though device deinit and
completion actions can still use OMPT state. The target order is now:

```
close runtime admission
  → deinit plugins and execute all measurement completion actions
  → assert OmptProfilerTy::ProfilerData.empty()
  → flush and drain all ready trace records
  → stop/join trace-buffer helper threads
  → destroy profiler and remaining runtime resources
```

`Plugin.release()` → real deletion remains a separate follow-up. Plugin deletion
exposes lifetime issues beyond profiler ownership, especially after partial
plugin-deinit failure, and should land only after all backend teardown paths
establish their own resource-lifetime postconditions.

### 8.4 Descriptor unregister: flush is not final shutdown

`__tgt_unregister_lib()` is descriptor/DSO-scoped. Compiler-generated wrapper
code invokes it when an offload-containing host image unregisters; other
registered libraries may remain and the runtime reference count may stay
nonzero. Therefore `PluginManager::unregisterLib()` must not tear down the
process-wide OMPT trace infrastructure.

The current call to `flushAndShutdownHelperThreads()` in `unregisterLib()` must
be split:

- `unregisterLib(Desc)` flushes all records ready at that boundary and waits for
  their buffer-completion callbacks, but leaves helper threads running. This
  happens before descriptor translation/host-entry state is removed so records
  carrying addresses associated with the unloading DSO are delivered in time.
- Final `OmptProfilerTy::deinit()` flushes remaining records and then shuts down
  and joins helper threads.

The TBM API should express both postconditions, for example a reusable
`flushAllBuffers`/`flush` operation and a final `flushAndShutdown` operation.
After a descriptor flush, tracing remains usable; after final shutdown, no ready
record, outstanding flush callback, or helper thread remains.

### 8.5 Prior art / related fixes found in history

- `26deaa6e03ac`/`749aac68332b` — "Wait for any outstanding flushes to
  complete during shutdown" (the `waitForFlushCompletion()` branch in
  `flushAndShutdownHelperThreads()`).
- `6a15e2d75623` — "Fix ordering with RPC teardown and global
  destructors" — same *class* of bug (teardown ordering), different
  subsystem.
- `a1d93f19bed4` — "Fix use-after-free of the packed firstprivate
  transfer buffer" — directly analogous bug shape (async AMDGPU op
  referencing host-side data freed before the op completed), fixed by
  extending the owning object's lifetime to match the async operation.
  This is precedent for the pattern §7/§8.3 propose.
- `376874a345d3` — "Fix destroying signal that was never initialized" —
  another signal-lifetime bug in the same plugin.
- No commit specifically addresses a profiler lifetime/UAF issue, which
  is consistent with §8.2: the current design never had one, by
  accident.

## 9. Design decisions and rationale (condensed)

| Decision | Rationale |
|---|---|
| Owner = `PluginManager`, not `GenericPluginTy` | Only `PluginManager` sits above all plugins and is where "one profiler shared across backends" naturally lives. |
| Reference for sync API, pointer for async-captured state | Matches the existing `void *ProfilerSpecificData` / `TracerInterfaceRAII` convention; avoids inventing a second ownership idiom. |
| No-op profiler always constructed, even without OMPT | Preserves "never null" call convention; today's `GenericProfilerTy` defaults already assume this. |
| Don't change virtual `dataSubmitImpl`/`Retrieve`/`Exchange` signatures | Keeps CUDA/host/level_zero untouched; the profiler already flows through `ProfilingInfoTy`/`AsyncInfo` for AMDGPU without needing it in the virtual signature. |
| Don't remove AMDGPU's `#ifdef OMPT_SUPPORT` gates now | Real cleanup, but orthogonal to ownership — bundling it risks conflating two reviews. Tracked as follow-up. |
| `liboffload` gets its own no-op profiler, not access to `PM` | `liboffload` doesn't link `libomptarget`; giving it a separate, equally-explicit profiler source is consistent with the whole point of the refactor (explicit, pluggable profiler per consumer). |
| `OmptProfilerTy` owns `OmptTracingBufferMgr` and its measurement map | Measurement payloads point into TBM-managed records; one OMPT-specific owner can enforce their relative lifetime and final delivery. No separate session abstraction is needed now. |
| Plugin `deinit()` establishes measurement completion | Backends own queues and slot actions. A successful deinit must execute those actions; `OmptProfilerTy::deinit()` asserts the measurement map is already empty rather than waiting for backend progress. |
| Tracing enabled/disabled state is fixed during execution | This keeps map membership equivalent to outstanding accepted measurements. Unsupported state changes while work is in flight are encoded with assertions. |
| Descriptor unregister flushes but does not shut down TBM | `__tgt_unregister_lib()` may run while other offload DSOs remain registered. Final helper-thread shutdown belongs only to `OmptProfilerTy::deinit()`. |
| `PluginManager` uses only generic profiler lifecycle | It owns the selected profiler and calls `Profiler->deinit()` after plugins deinitialize; OMPT-specific waiting, flushing, and helper-thread teardown remain encapsulated. |

## 10. Patch sequencing

Recommended as separate, independently reviewable patches, in this order:

1. **Split descriptor flush from final TBM shutdown.** Change
   `PluginManager::unregisterLib()` to flush and wait for currently ready
   records without stopping helper threads (§8.4). Preserve the existing
   pre-descriptor-removal ordering.
2. **Encode backend deinit postconditions.** Ensure successful plugin/device
   deinitialization drains backend-owned queues and executes every associated
   slot action. Add AMDGPU assertions for pending stream slots/actions and make
   every terminal submission/error path retire any allocated profiler-data
   entry (§7.2, §8.3).
3. **Introduce `PluginManager`-owned `Profiler`; remove
   `getProfilerToAttach()` weak/strong pair and `GenericPluginTy::Profiler`.**
   Thread the parameter through every function in §5.1 and its AMDGPU
   overrides (§5.2), updating all call sites down to the STOP points. Update the
   four already-explicit libomptarget sites (§5.4) to source from
   `PM->getProfiler()`.
4. **Move `OmptTracingBufferMgr` under `OmptProfilerTy` and add generic
   profiler deinitialization.** After all plugins deinitialize,
   `PluginManager` calls `Profiler->deinit()`. The OMPT override asserts an
   empty measurement map, performs the final TBM flush, waits for delivery,
   and stops/joins helper threads. The no-op profiler returns immediately.
5. **Update `liboffload` in the coordinated common-interface change:** give it
   its own no-op profiler instance and update its call sites (§5.5), keeping
   every in-tree consumer buildable at each landed revision.
6. **(Deferred, separate series)** Convert `Plugin.release()` to real deletion
   only after partial-failure and backend-resource teardown are independently
   proven safe.
7. **(Deferred, separate series)** Clean up AMDGPU's `#ifdef OMPT_SUPPORT`
   gates in `rtl.cpp` now that the profiler abstraction is fully generic.

## 11. Test strategy

- Existing OMPT device-tracing tests under `offload/test` must continue to pass
  for both OMPT-enabled and OMPT-disabled builds.
- Add focused assertions/tests for the plugin/profiler boundary: after
  successful plugin deinit, `OmptProfilerTy::ProfilerData` is empty; profiler
  deinit rejects any new measurement; and every submission failure/slot
  rollback retires its allocated entry.
- Add a multi-library test covering the descriptor/runtime distinction:
  register two offload descriptors, unregister one, verify that ready records
  are flushed but TBM helpers and tracing remain usable for the second, then
  unregister the last and verify final profiler/TBM shutdown.
- Verify final ordering with an async kernel/copy: backend teardown executes its
  profiling slot action, map emptiness is observed before profiler deinit,
  and the corresponding ready record is delivered during the final TBM flush.
- Add/retain coverage for failed HSA timestamp queries. The existing
  zero-initialized `{0, 0}` timing record must remain structurally decodable,
  be marked ready, and retire its measurement entry rather than blocking
  shutdown.
- Run the AMDGPU offload tests under ASan/TSan where supported because the
  changes touch stream-slot, callback, profiler-payload, and helper-thread
  lifetimes.
- Run the AOMP repository's `smoke`, `smoke-limbo`, and `smoke-dev` test suites.
  These provide additional integration coverage, particularly for OMPT device
  tracing and its runtime initialization, completion, flush, and teardown paths.
- CUDA/host/level_zero require coordinated build and existing-suite coverage;
  §5.3 avoids changing their profiling implementation surface.

## 12. Open questions for reviewer

The following implementation details remain open; the ownership and lifecycle
choices above are settled for this plan.

1. What is the safe failure policy if plugin deinit cannot drain a backend or
   the profiler map is unexpectedly nonempty? Logging and destroying referenced
   state is not valid. Options are a fatal error, intentional retention of the
   affected resources, or a narrowly defined cancellation path.
2. Which precise runtime mechanism enforces the assumption that no new offload
   operation can begin once final runtime deinitialization starts? The existing
   `RTLAlive`/`RTLOngoingSyncs` behavior must be checked against all submission
   paths, even though profiler code does not own this admission barrier.
3. Which TBM state should be exposed for strong debug postconditions after
   descriptor flush and final shutdown (no outstanding flush IDs, no
   undelivered ready records, no helper threads)?
4. Is a single global `PluginManager`-owned profiler sufficient, or does any
   near-term use case need per-device/per-plugin instances? The current scope
   chooses one per libomptarget runtime, but per-device lifecycle and clock
   conversion state inside `OmptProfilerTy` must still be represented at the
   correct granularity.
5. For `liboffload`'s no-op profiler (§5.5): singleton, or one per
   `ol_context`/`PluginContextTy`? Prefer matching the lifetime of the plugin
   instances used by that consumer.
6. The plan defers `Plugin.release()` removal. When that follow-up is attempted,
   how should partial plugin/device deinit errors be accumulated so later
   devices are still drained before object destruction?
