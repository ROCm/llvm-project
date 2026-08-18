# Review of `ProfilerOwnershipRefactor.md`

## Overall assessment

The proposal identifies a real architectural problem: selecting the profiler through a weak/strong symbol pair is surprising, prevents explicit composition, and makes ownership depend on final link composition. The inventory of current call sites is useful, and the discovery that shutdown safety currently depends on leaked plugin objects is important.

I do **not** think the proposal is ready to implement as written. The ownership direction is reasonable, but the selected parameter-threading design is substantially larger than necessary, the proposed shutdown drain cannot be implemented using the API cited in the document, and moving to one process-wide `OmptProfilerTy` changes observable OMPT device-lifecycle behavior.

The teardown work should be separated from the ownership refactor and designed first. It is independently valuable and is a prerequisite for giving any currently leaked object a finite lifetime.

## Blocking concerns

### 1. The proposed shutdown drain has no implementable path

The safety argument depends on this statement:

> For every initialized plugin, for every live device, call the existing `GenericDeviceTy::synchronize`/device-level synchronize path.

That is not an available “synchronize this device” operation. `GenericDeviceTy::synchronize` requires a specific `__tgt_async_info *` and only synchronizes the queue stored in that object (`plugins-nextgen/common/src/PluginInterface.cpp:1048-1069`). The AMDGPU implementation likewise extracts exactly one `AMDGPUStreamTy *` from `AsyncInfo.Queue` (`plugins-nextgen/amdgpu/src/rtl.cpp:4016-4031`).

`PluginManager` owns the `DeviceTy` wrappers (`include/PluginManager.h:44-50,184-187`), but it does not track all live `AsyncInfoTy` objects or their queue handles. `DeviceTy::synchronize` still requires such an object (`libomptarget/device.cpp:450-451`). The AMDGPU stream manager also does not expose a “drain all checked-out streams” operation; its `deinit()` deinitializes queues and then lets the generic resource manager warn about resources not returned (`plugins-nextgen/amdgpu/src/rtl.cpp:3021-3030`, `plugins-nextgen/common/include/PluginInterface.h:2264-2277`).

Therefore §8.3 and patch 1 in §10 are not actionable as stated. Before implementation, the design must choose and specify one of the following kinds of mechanism:

- runtime-wide tracking of every outstanding async queue, with a defined ownership and locking protocol;
- a backend/device `drain()` operation that safely finds and waits for every checked-out stream;
- a shutdown admission protocol that prevents new work, waits for all active runtime operations, and then drains backend-owned work.

This is not a small detail. It is the operation on which the raw-pointer lifetime proof rests.

### 2. The proposed trace shutdown order is backwards for asynchronous producers

The document recommends:

> flush/shutdown trace threads → synchronize all devices → deinit plugins

AMDGPU completion processing calls `OmptProfilerTy::handleKernelCompletion` / `handleDataTransfer`, which completes OMPT trace records (`plugins-nextgen/common/OMPT/OmptProfiler.cpp:95-145`). Those completion calls occur while a stream is completed (`plugins-nextgen/amdgpu/src/rtl.cpp:2047-2079`). Thus device synchronization is a **producer** of final trace records.

Shutting down the trace-buffer helper threads before draining devices permits the drain itself to publish records after the consumer side has stopped. This is especially concerning because `PluginManager::unregisterLib()` already calls `flushAndShutdownHelperThreads()` before runtime deinitialization (`libomptarget/PluginManager.cpp:393-400`). The proposal correctly notes that the current flush is best-effort, but then preserves the same producer/consumer inversion in its recommended order.

There is also a concrete payload-lifetime failure if tracing is disabled before completion: both `handleKernelCompletion()` and `handleDataTransfer()` return immediately when `isProfilingEnabled()` is false, before calling `freeProfilerDataEntry()` (`plugins-nextgen/common/OMPT/OmptProfiler.cpp:95-119,122-144`). An operation that captured profiler data while tracing was active can therefore retain its `OmptEventInfoTy` indefinitely if tracing becomes inactive before the stream is drained. The shutdown design must separate “accept new trace events” from “finish and release already accepted events,” or otherwise provide explicit cancellation cleanup.

The required ordering should be reasoned about explicitly, but at minimum it needs the following phases:

1. prevent admission of new offload/profiling work;
2. drain all operations that can produce profiler callbacks;
3. flush all records produced by that drain and stop trace helper threads;
4. issue device-finalize callbacks while the profiler and any OMPT state they need remain valid;
5. destroy tracing state, profiler, devices, and plugins in a proven order.

There is an additional design question here: if `handleDeinit()` is itself allowed to produce tracing work, trace shutdown must occur after device deinitialization rather than before it. The proposal currently assumes that today’s implementation detail (“it happens not to touch the manager”) is enough, while simultaneously arguing that implementation accidents should not carry lifetime correctness.

### 3. A single `OmptProfilerTy` changes device lifecycle semantics

The proposal calls the refactor behavior-preserving, but `OmptProfilerTy` contains profiler-instance state that is currently effectively per plugin. In particular, `handleInit()` emits `device_initialize` only when its `OmptInitialized` member transitions from false to true, and `handleDeinit()` emits `device_finalize` only when it transitions from true to false (`plugins-nextgen/common/OMPT/OmptProfiler.cpp:20-47`; member declared in `plugins-nextgen/common/OMPT/OmptProfiler.h:156`).

Today there is one profiler per plugin, so this state suppresses callbacks after the first initialized device **within each backend**. With one process-wide profiler, it suppresses callbacks after the first initialized device **across every backend**. The first device deinitialized also flips the shared state to false, so later devices may not receive a matching finalize callback. That is an observable semantic change and can produce mismatched lifecycle notifications in a heterogeneous process.

The proposal must define the desired lifecycle granularity and update the profiler state model accordingly. If OMPT requires one initialize/finalize callback per device, this likely needs per-device state rather than one boolean. If current per-plugin behavior is intentional, one global profiler cannot preserve it without keying state by plugin. Either way, this must be resolved before selecting the owner.

The same consolidation also changes `ProfilerData` from one map/mutex per plugin to one process-wide map/mutex (`plugins-nextgen/common/OMPT/OmptProfiler.h:114-143`). That may be acceptable, but it is a behavior and contention change, not merely ownership plumbing.

Clock correlation has the same unresolved granularity problem. `setTimeConversionFactors()` stores mutable slope/offset state on the profiler (`plugins-nextgen/common/include/GenericProfiler.h:65-72,177-181`), while the OMPT implementation ultimately updates process-wide conversion state. Sharing the profiler across plugins and devices allows later initialization/calibration to overwrite earlier state. The design must establish whether conversion is per process, backend, or device and key the state accordingly.

### 4. Parameter threading is not justified against simpler dependency injection

The proposed signature changes push `GenericProfilerTy &` through a large portion of the common plugin API, including methods whose profiler dependency is incidental. This permanently adds profiler plumbing to every caller and makes future API changes repeat the same exercise.

A materially smaller design satisfies the stated ownership goals:

- the consumer (`PluginManager` for libomptarget, a liboffload context/platform for liboffload) owns a profiler;
- `GenericPluginTy` stores a non-owning, non-null profiler pointer/reference initialized by the consumer before plugin/device initialization;
- existing `Plugin.getProfiler()` access remains, but no plugin constructs or owns the object;
- the weak/strong `getProfilerToAttach()` pair is removed.

This still makes ownership and selection explicit, permits one instance to be shared across plugins, and gives liboffload an independent profiler. Its async lifetime requirements are exactly the same as those of the proposal: a captured pointer remains valid until the consumer drains and tears down the plugin.

The review document should not assume this alternative is necessarily the final answer, but the implementation plan must compare against it. “References document non-nullness” is not enough benefit to justify the much larger signature surface in §5. If direct parameter passing is required to prevent plugins from having any ambient profiler dependency, that architectural constraint should be stated explicitly and defended.

### 5. Shutdown does not yet prevent concurrent runtime work

`deinitRuntime()` sets `RTLAlive = false`, but the only active-operation counter it waits on is `RTLOngoingSyncs`, which covers explicit target synchronization (`libomptarget/OffloadRTL.cpp:64-79`; uses in `libomptarget/OpenMP/API.cpp:839-850`). It is not a general count of in-progress offload calls, callbacks, queue submissions, or async task completion.

Even a correct backend drain does not by itself prove that no other host thread can be entering or executing a path that uses the profiler. The lifetime design needs an admission and quiescence protocol, not only a device wait. It must answer:

- Which public/internal entry points are rejected after shutdown starts?
- How are calls that passed the admission check before shutdown counted?
- Can OMPT callbacks re-enter libomptarget during device synchronization or finalization?
- Which locks may be held while invoking tool callbacks?
- How is a race between new queue submission and enumeration/drain of queues prevented?

Until those questions have concrete answers, a raw profiler pointer captured into asynchronous state is not “safe by construction.”

### 6. An HSA signal wait is not yet proven to join the async handler

The proposal notes that `hsa_amd_signal_async_handler` stores a raw pointer into a stream’s `Slots` deque, then claims a device synchronize makes teardown safe. The AMDGPU code shows that `asyncActionCallback()` executes `Slot->performAction()`, signals the output signal, and only then returns `false` to unregister (`plugins-nextgen/amdgpu/src/rtl.cpp:2117-2139`).

The plan needs evidence that the proposed drain does not merely observe the output signal while the handler is still between `signal()` and callback return/unregistration. If HSA guarantees callback completion/unregistration before a waiter can proceed, cite that contract. If it does not, shutdown needs an explicit callback-in-flight count, deregistration/join operation, or another lifetime mechanism for callback arguments. This matters for plugin/device deletion even if profiler callbacks never run on the handler thread today.

## Major recommendations

### Separate teardown correctness from profiler ownership

I agree with the proposed sequencing direction, but patch 1 needs a real design before it can be called independently testable. Treat it as a separate shutdown project with explicit invariants:

- no new operations can start;
- all admitted host operations have exited or reached a safe state;
- every backend queue and callback is drained;
- no trace producers remain before trace consumers are stopped;
- deinitialization errors do not skip cleanup of later devices/plugins;
- the order remains safe when callbacks re-enter or tracing is disabled.

Only after those invariants are implemented should the profiler acquire a finite lifetime.

### Keep plugin deletion out of this series

Changing `Plugin.release()` to normal `unique_ptr` destruction is desirable, but it exposes every lifetime bug currently hidden by the leak, not only profiler lifetime bugs. The current plugin deinit path can return on the first device error (`plugins-nextgen/common/src/PluginInterface.cpp:1509-1519`), leaving later devices active. Deleting the plugin after such a partial failure is a substantially broader change.

Make plugin deletion a follow-up after backend drains and error-accumulating teardown are independently established and tested. Do not present it as an optional mechanical final step of the ownership work.

### Specify construction timing

`OmptProfilerTy` binds callback and tracing function pointers in its constructor, conditional on `ompt::Initialized` (`plugins-nextgen/common/OMPT/OmptProfiler.h:56-83`). Today plugin construction happens during `PluginManager::init()` after `connectLibrary()` is called from runtime initialization (`libomptarget/OffloadRTL.cpp:50-57`, `libomptarget/PluginManager.cpp:45-60`).

The new owner must construct `OmptProfilerTy` at a point with the same or stronger ordering guarantee. This should be stated as an invariant, not left as “constructed once in `PluginManager::init()`.” Also define behavior if OMPT connection fails or changes state after construction.

### Avoid claiming ABI analysis is complete without checking all static consumers

The analysis correctly identifies libomptarget and liboffload as static consumers. However, the implementation should use build-system dependency enumeration rather than assuming those are the only consumers. A signature-threading design has a much larger exposure to out-of-tree/static consumers than injection with stable call signatures. This is another reason to minimize common API churn unless it buys a clear property.

## Test strategy concerns

The current test strategy is too aspirational to validate the lifetime proof.

1. A lit test that starts an async copy and exits without a wait may be invalid at the OpenMP semantic level or may be made synchronous by task/runtime behavior. The test must demonstrate that work is actually outstanding when shutdown starts.
2. “Run the full AMDGPU suite under ASan/TSan” is useful validation, but it is not a regression test. GPU runtime and system-library instrumentation limitations also need to be acknowledged.
3. There is no proposed test for multiple plugins/devices, which is essential because profiler consolidation changes `OmptInitialized` behavior.
4. There is no test for a deinit error partway through multiple devices/plugins.
5. There is no test that proves trace records produced during drain are flushed before helper-thread shutdown.
6. There is no test for concurrent submission racing with shutdown or OMPT callback reentrancy.
7. OMPT tests currently register device lifecycle callbacks, but the proposal should add checks that every initialized device receives a correctly paired finalization callback in multi-device and, where available, heterogeneous configurations.

The design would benefit from an injectable fake profiler and fake backend/queue in a unit test. Such a test could deterministically block completion, start shutdown, verify object liveness and callback ordering, then release completion. Hardware-only tests should supplement that deterministic test rather than serve as the sole proof.

## Design questions and clarifications

These should be resolved in the design document before implementation:

1. What concrete object tracks or can enumerate every outstanding asynchronous queue in libomptarget?
2. What exact API performs a device-wide/backend-wide drain when no `AsyncInfoTy` is available?
3. What synchronization prevents a new queue from being submitted while shutdown enumerates queues?
4. Does an HSA signal wait guarantee the associated async handler has returned and unregistered? Where is that contract documented?
5. Should trace helper threads stop before or after `handleDeinit()`? Can current or future `handleDeinit()` implementations publish trace records?
6. Why is passing a profiler through every synchronous API preferable to injecting a non-owning profiler into each plugin at initialization?
7. Is one profiler per process a hard requirement, or merely the current desired configuration? The document alternates between “explicit profiler per consumer” and “exactly one for the whole process.”
8. What is the required OMPT `device_initialize`/`device_finalize` granularity, and how will one profiler represent state for multiple devices and plugins?
9. Which component owns liboffload’s profiler, and what lifetime relation does it have to `ol_platform_impl_t`, `PluginContextTy`, devices, and outstanding queues? Leaving singleton versus per-context unresolved means the plan is not implementation-ready.
10. What happens to outstanding `OmptEventInfoTy` entries when an operation fails before scheduling a completion action, queue submission is rolled back, synchronization fails, or shutdown continues after a best-effort drain error?
11. If drain is best-effort and fails, how can destruction safely continue? Logging and continuing is incompatible with a lifetime proof unless the affected objects are intentionally retained/leaked.
12. Is profiler construction guaranteed to happen after OMPT callback lookup is initialized? How is failed/late OMPT initialization handled?
13. Are profiler methods expected to be callable concurrently across devices and backends? If so, which fields are protected? `setTimeConversionFactors()` writes ordinary `double` members (`plugins-nextgen/common/include/GenericProfiler.h:65-72,177-181`) and will become shared across plugins.
14. Are time-conversion factors actually per process, per backend, or per device? A single profiler lets one AMDGPU device/backend overwrite factors derived by another.
15. Is the goal to remove ownership from `GenericPluginTy`, or to remove all profiler awareness from it? Those are different architectural goals and lead to different API designs.

## Suggested revised direction

1. Design and land a standalone shutdown/quiescence change with a real all-queue drain API and deterministic tests.
2. Correct trace producer/consumer shutdown ordering.
3. Define profiler state granularity (process, plugin, device), especially OMPT lifecycle and clock conversion state.
4. Compare explicit plugin injection against full parameter threading and select the smaller mechanism that satisfies a stated architectural invariant.
5. Move ownership and remove `getProfilerToAttach()` only after the lifetime foundation is in place.
6. Add liboffload integration as part of the ownership patch that changes the common interface, not as a later patch during which the tree may not build.
7. Defer actual plugin destruction until partial-failure teardown and all backend callback lifetimes are independently safe.

## Conclusion

The document is strong as an investigation report but not yet as an implementation plan. Its most valuable finding is the pre-existing shutdown/lifetime debt. The main refactor should not proceed until there is an implementable device-wide drain, correct trace shutdown ordering, and an explicit answer for shared `OmptProfilerTy` state. After that, the ownership move should be made with the least invasive dependency-injection mechanism that meets the actual consumer requirements.
