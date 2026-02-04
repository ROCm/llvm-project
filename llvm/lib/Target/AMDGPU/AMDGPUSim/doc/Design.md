# AMDGPUSim Library Design

## Overview

AMDGPUSim is a standalone AMDGPU static performance simulator library. It provides cycle-accurate simulation of instruction execution, tracking stalls from various sources including functional unit contention, memory latencies, WMMA co-execution rules, and register bank conflicts.

The library is designed to be used by:
- **MachineFunction passes** (via `MachineInstrInfo` adapter)
- **MC layer passes** (via `MCInstInfo` adapter)
- **External tools** (implementing custom `SimInstInfo`)

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         User Code (Pass/Tool)                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │ MachineInstrInfo │              │    MCInstInfo    │            │
│  │  (MIR Adapter)   │              │  (MC Adapter)    │            │
│  └────────┬─────────┘              └────────┬─────────┘            │
│           │                                  │                     │
│           │  implements SimInstInfo          │                     │
│           └──────────────┬───────────────────┘                     │
│                          ▼                                         │
│              ┌───────────────────────┐                             │
│              │     SimInstInfo       │ ◄── Abstract Interface      │
│              │  (property queries)   │                             │
│              └───────────┬───────────┘                             │
│                          │                                         │
│                          ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Simulator Class                          │  │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │  │
│  │  │ SimInst │  │ HWModel  │  │ SimState  │  │ simulateInst │  │  │
│  │  └─────────┘  └──────────┘  └───────────┘  └──────────────┘  │  │
│  │                    ↕ owns GPUSimState                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Library (`LLVMAMDGPUSim`)

The core library contains the simulator logic with no dependencies on LLVM CodeGen or MC layers.

#### `SimInst` (SimInst.h)

Lightweight instruction wrapper holding:
- `void* InstPtr` - Pointer to actual instruction (MachineInstr*, MCInst*, etc.)
- `InstClass Class` - Cached instruction classification
- `unsigned Latency` - Cached latency
- `FunctionalUnit Unit` - Cached functional unit

Also defines enums used throughout:
- `InstClass` - Instruction classification (VALU, SALU, TRANS, WMMA, DS_READ, etc.)
- `FunctionalUnit` - Hardware units (XDL, VALU, SALU, TRANS, LDS, VMEM, SMEM, BRANCH)
- `WMMAStageType` - WMMA co-execution window stages (E0, E, I, V)
- `WaitType` - Wait instruction types (DS, VMEMLoad, VMEMStore, SMEM, Tensor, DepCtr, etc.)
- `WMMAVariant` - WMMA instruction variants for co-execution rules
- `RegOperand` - Register operand info for bank conflict analysis

#### `SimInstInfo` (SimInstInfo.h)

Abstract interface for querying instruction properties. Concrete implementations provide the actual property extraction from the underlying instruction type.

```cpp
class SimInstInfo {
public:
  // VALU/TRANS
  virtual unsigned getRepeatRate(const SimInst &SI) const = 0;
  virtual bool isLOLVALU(const SimInst &SI) const = 0;
  virtual bool isTRANS(const SimInst &SI) const = 0;
  virtual unsigned getResourceCycles(const SimInst &SI) const = 0;

  // delay_alu / Wait
  virtual unsigned getDelayAluImm(const SimInst &SI) const = 0;
  virtual std::pair<WaitType, unsigned> getWaitInfo(const SimInst &SI) const = 0;
  virtual unsigned getVaVdstTarget(const SimInst &SI) const = 0;

  // Memory
  virtual std::pair<unsigned, unsigned> getDestRegInfo(const SimInst &SI, bool IsVGPR) const = 0;

  // WMMA
  virtual WMMAVariant getWMMAVariant(const SimInst &SI) const = 0;
  virtual bool hasScaling(const SimInst &SI) const = 0;

  // Registers
  virtual bool hasSGPROperands(const SimInst &SI) const = 0;
  virtual void getSrcRegs(const SimInst &SI, SmallVectorImpl<RegOperand> &Regs) const = 0;
  virtual void getWMMASrcRegs(const SimInst &SI, SmallVectorImpl<RegOperand> &Regs) const = 0;
  virtual void getDstRegs(const SimInst &SI, SmallVectorImpl<RegOperand> &Regs) const = 0;

  // Counting / Classification
  virtual bool isVOPD(const SimInst &SI) const = 0;
  virtual bool isPacked(const SimInst &SI) const = 0;
  virtual bool waitsForVALU(const SimInst &SI) const = 0;
  virtual unsigned getInstBytes(const SimInst &SI) const = 0;
};
```

Key notes:
- `getSrcRegs()` returns all explicit uses with non-register placeholders to preserve port indexing for the VGPR source cache.
- `getWMMASrcRegs()` returns only the A and B matrix VGPR sources (src0, src1), skipping C (tied-def) and scale registers. Used for WMMA-specific cache tracking.
- `waitsForVALU()` indicates instructions that implicitly wait for all pending VALU operations (VA_VDST==0), used for scoreboard clearing.

#### `HWModel` (HWModel.h)

Hardware configuration parameters including:
- Default latencies for instruction classes
- WMMA co-execution window configurations
- Memory FIFO depths
- VA_VDST multiplier
- IS cache parameters (`ISCacheNumLines`, `ISCacheLineSize`, `SQCToISLatency`)

Factory: `HWModel::gfx1250()`

#### `GPUSimState` (SimState.h)

Simulation state including:
- Current cycle counter
- Functional unit busy times
- Pending memory operations (DS, VMEM, SMEM, TDM)
- Recent VALU/TRANS tracking for delay_alu
- Active WMMA window state
- Register file with VGPR source cache
- Register scoreboard for RAW hazard tracking
- **Instruction Store (IS) cache state** (`ISCacheState`) — models the L0 instruction cache, tracking line consumption, fetch triggers, and fetch latency stalls

The `GPUSimState` is owned by the `Simulator` class and is not designed to be subclassed by users. Users can inspect it via `Simulator::getState()` which returns a `const GPUSimState &`.

#### `InstrSimInfo` (InstrInfo.h)

Per-instruction simulation result:
- `StallCycles` - Total stall cycles
- `Reason` - Dominant stall reason (enum `StallReason`)
- `Breakdown` - Detailed per-source stall breakdown (`StallBreakdown` struct)
- WMMA window info (stage, co-execution status)
- Register bank stall info
- Cache pattern info
- MSB_SET fusion/exposure/masking flags
- **IS cache results** (when `EnableISCache` is set):
  - `ISFetchStall` — total IS cache stall cycles for this instruction
  - `ISFetchesTriggered` — number of new cache line fetches triggered
  - `ISBytesConsumed` — instruction size in bytes consumed from the IS cache

##### `StallBreakdown`

Detailed stall breakdown populated by the simulator core, used by the pass for verbose logging and `BlockMetrics` attribution:

```cpp
struct StallBreakdown {
  unsigned FU, VALUSlot, CoExec, CoExecFromEffective, EffectiveCycle;
  bool HasFUCoExecInteraction;
  unsigned DelayAlu, WaitCnt, MemFIFO, RegBank;
  unsigned LongLatVALU, LOLVALUTRANSHazard, SSRC, VaVdst, RAW;
  unsigned ISFetch;
  bool RegBankInWMMAWindow;
  bool IsScaledWMMA;
  unsigned WMMAStartCycle;

  unsigned total() const;    // Max across all sources (RegBank excluded if in WMMA window)
};
```

#### `Simulator` Class (Simulator.h)

The `Simulator` class wraps the core simulation logic and owns the `GPUSimState`, `SimInstInfo` reference, `HWModel` reference, and configuration.

```cpp
struct SimulatorConfig {
  bool Verbose = false;          // Enable verbose logging
  raw_ostream *Log = nullptr;    // Output stream for verbose logging
  bool EnableScoreboard = false; // Enable RAW hazard tracking
  bool EnableISCache = false;    // Enable IS cache modeling
};

class Simulator {
  const SimInstInfo &InstInfo;
  const HWModel &Model;
  GPUSimState State;
  SimulatorConfig Config;

public:
  Simulator(const SimInstInfo &II, const HWModel &M, SimulatorConfig C = {});

  InstrSimInfo simulateInst(const SimInst &Inst, ArrayRef<SimInst> Lookahead = {});
  void reset();
  void advanceCycles(unsigned N);
  const GPUSimState &getState() const;
  const SimulatorConfig &getConfig() const;
  const HWModel &getModel() const;
};
```

By wrapping state ownership and configuration in the `Simulator` class, `simulateInst()` only needs the instruction itself and an optional lookahead buffer. Verbose logging (stall breakdown, WMMA window info, unit state, delay_alu/va_vdst decode, RAW dependencies, cycle advances, IS cache events) is handled internally by the simulator when `Config.Verbose` is true. IS cache modeling is integrated into `simulateInst()` — when `EnableISCache` is set, the simulator applies IS pre-stalls (waiting for the current cache line to be ready) and post-stalls (line transition after byte consumption) around the core simulation, and populates `InstrSimInfo::ISFetchStall`, `ISFetchesTriggered`, and `ISBytesConsumed`.

#### `Simulator.cpp`

Core simulation logic implementing:
1. `computeStallSources()` - Computes all stall sources (unit busy, co-exec, delay_alu, va_vdst, wait, RAW, reg bank, etc.)
2. `populateInstrSimInfo()` - Fills `InstrSimInfo` and `StallBreakdown` from stall sources
3. `recordInstruction()` - Records instruction effects on state (unit busy, WMMA window, pending ops, scoreboard)
4. IS cache integration: pre-stall (current line ready wait), post-stall (line transition after byte consumption)
5. Verbose logging: stall breakdown, delay_alu/va_vdst decode, RAW dependency info, WMMA window/occupancy, unit state, cycle advance, IS fetch events

### MIR Adapter (`LLVMAMDGPUSimMIRAdapter`)

#### `MachineInstrInfo` (MIRAdapter.h/.cpp)

Concrete `SimInstInfo` implementation for `MachineInstr`:
- Uses `SIInstrInfo` and `SIRegisterInfo` for property extraction
- Full access to scheduling model for accurate latencies and resource cycles
- Complete register operand information including named operands for WMMA

```cpp
MachineInstrInfo InstInfo(TII, TRI);
HWModel Model = createHWModel(GPUTarget::GFX1250);
SimulatorConfig Cfg;
Cfg.Verbose = true;
Cfg.Log = &dbgs();
Cfg.EnableScoreboard = true;

Simulator Sim(InstInfo, Model, Cfg);
for (auto &MI : MBB) {
  SimInst SI = InstInfo.createSimInst(MI);
  InstrSimInfo Result = Sim.simulateInst(SI);
}
Sim.reset();
```

### MC Adapter (`LLVMAMDGPUSimMCAdapter`)

#### `MCInstInfo` (MCAdapter.h/.cpp)

Concrete `SimInstInfo` implementation for `MCInst`:
- Uses `MCInstrInfo` and `MCRegisterInfo`
- Some features limited compared to MIR:
  - `isLOLVALU()` returns false (no repeat rate info)
  - `getRepeatRate()` returns a constant heuristic (TODO: tune later)
  - `getDestRegInfo()` returns conservative defaults
  - `getWMMASrcRegs()` falls back to `getSrcRegs()` (no named operand support)
  - Register operand info is simplified

```cpp
MCInstInfo InstInfo(MCII, MRI);
Simulator Sim(InstInfo, Model);
for (auto &MCI : Instructions) {
  SimInst SI = InstInfo.createSimInst(MCI);
  InstrSimInfo Result = Sim.simulateInst(SI);
}
```

## Stall Reasons

The simulator tracks these stall sources:

| Reason | Description |
|--------|-------------|
| `FU_BUSY` | Functional unit not available |
| `COEXEC_BLOCKED` | WMMA co-execution rules block instruction |
| `LONG_LAT_VALU` | Long-latency VALU can't co-execute with WMMA |
| `LOLVALU_TRANS_HAZARD` | 1-cycle mutual exclusion between LOLVALU and TRANS |
| `VA_SSRC_STALL` | VALU with SGPR operands blocks SALU |
| `VA_VDST_WAIT` | Waiting for VA_VDST counter |
| `RAW_HAZARD` | Read-after-write hazard (scoreboard) |
| `WAITCNT` | Explicit wait instruction |
| `DELAY_ALU` | delay_alu dependency |
| `MEM_FIFO` | Memory FIFO full |
| `MSB_SET_EXPOSED` | s_set_vgpr_msb not fused |
| `REG_BANK` | Register bank conflict |
| `IS_FETCH` | Instruction Store cache miss (modeled by library when `EnableISCache` is set) |

## Verbose Logging

When `SimulatorConfig::Verbose` is true and `Log` is non-null, the simulator outputs detailed per-instruction information including:

- **Stall breakdown**: `Stalls: FU=1, WMMACoExecMiss=2 → Total: 3 Cache($$-)`
- **delay_alu decode**: `DelayALU: instid0=1 (stall 4), skip=1, instid1=1 (pending)`
- **Pending instid1**: `PendingInstId1: Dep=1 stall=4`
- **va_vdst wait**: `s_wait_alu: va_vdst(0), pending=1, stall=19`
- **RAW dependency**: `RAW dependency: stall=4`
- **Cycle advance**: `→ Advancing cycle: 1 → 5`
- **WMMA window stage**: `WMMA Window: [2/9] I (cycles 0-9)`
- **WMMA occupancy**: `Class: WMMA | Unit: XDL | Occupancy: 8 | Window: 9`
- **Active WMMA**: `→ ActiveWMMA: cycles 0-9 [back-to-back]`
- **Unit state**: `→ UnitBusyUntil[VALU] = 3`, `→ LastVALUCycle = 2`
- **Memory state**: `→ PendingDS: 2, Counter[LGKM]=2`
- **FU+CoExec interaction**: `(Base stall lands at cycle 5 [stage 4/9 E - blocked] → additional CoExec=2)`

- **IS fetch stall**: `IS fetch stall: line 2 not ready, stall=26`
- **IS line transition**: `IS line transition stall: +26 cycles`

The pass adds its own logging for:
- Instruction header (cycle, opcode, class, unit, latency, IS cache line state)
- MSB_SET fusion/exposure
- Block/function summaries

## Usage Example

```cpp
#include "AMDGPUSim/AMDGPUSim.h"
#include "AMDGPUSim/MIRAdapter.h"
#include "AMDGPUSim/Simulator.h"

using namespace llvm::AMDGPUSim;

void analyzeFunction(MachineFunction &MF) {
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  const SIRegisterInfo &TRI = TII.getRegisterInfo();

  HWModel Model = createHWModel(GPUTarget::GFX1250);
  MachineInstrInfo InstInfo(TII, TRI);

  SimulatorConfig Cfg;
  Cfg.Verbose = true;
  Cfg.Log = &dbgs();
  Cfg.EnableScoreboard = true;
  Cfg.EnableISCache = true;  // Enable IS cache modeling

  Simulator Sim(InstInfo, Model, Cfg);

  unsigned TotalCycles = 0;
  unsigned TotalStalls = 0;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      SimInst SI = InstInfo.createSimInst(MI);
      InstrSimInfo Result = Sim.simulateInst(SI);

      TotalCycles++;
      TotalStalls += Result.StallCycles;

      if (Result.Reason != StallReason::NONE) {
        // Attribute stalls via Result.Breakdown
      }
    }
  }

  // Access state for post-analysis (e.g., false-wait detection)
  const GPUSimState &State = Sim.getState();
  // ... inspect State.PendingDS, State.PendingVMEMLoad, etc.

  Sim.reset(); // For next function
}
```

## Build Configuration

The library is built as three separate components in `AMDGPUSim/CMakeLists.txt`:

```cmake
# Core simulation library (no CodeGen/MC dependencies)
add_llvm_component_library(LLVMAMDGPUSim
  Simulator.cpp
  LINK_COMPONENTS Support
)

# MIR adapter (depends on CodeGen)
add_llvm_component_library(LLVMAMDGPUSimMIRAdapter
  MIRAdapter.cpp
  LINK_COMPONENTS AMDGPUSim CodeGen Support
)

# MC adapter (depends on MC)
add_llvm_component_library(LLVMAMDGPUSimMCAdapter
  MCAdapter.cpp
  LINK_COMPONENTS AMDGPUSim MC Support
)
```

The `AMDGPUStaticSimulator` pass links `LLVMAMDGPUSimMIRAdapter` (which transitively links `LLVMAMDGPUSim`). The MC-layer pass links `LLVMAMDGPUSimMCAdapter`.

## Design Benefits

1. **Clean Separation** - Core simulator doesn't know about MachineInstr/MCInst
2. **Lazy Evaluation** - Properties computed on-demand, not upfront
3. **Extensibility** - Easy to add new instruction sources (e.g., external tools)
4. **Small SimInst** - Only caches frequently-used fields (Class, Latency, Unit)
5. **Minimal Dependencies** - Core library only depends on LLVM Support
6. **Encapsulated State** - `Simulator` class owns `GPUSimState`; users inspect via const ref
7. **Configurable Logging** - Verbose output controlled by `SimulatorConfig`, matching the exact format expected by existing tests

## File Structure

```
AMDGPUSim/
├── AMDGPUSim.h      # Main public header (includes core headers)
├── SimInst.h        # Lightweight instruction wrapper + enums
├── SimInstInfo.h    # Abstract property query interface
├── HWModel.h        # Hardware model configuration
├── SimState.h       # Simulation state (GPUSimState, RegisterFile, etc.)
├── InstrInfo.h      # Per-instruction result (InstrSimInfo, StallBreakdown, StallReason)
├── Simulator.h      # Simulator class and SimulatorConfig
├── Simulator.cpp    # Core simulation logic
├── MIRAdapter.h     # MachineInstrInfo declaration
├── MIRAdapter.cpp   # MachineInstrInfo implementation
├── MCAdapter.h      # MCInstInfo declaration
├── MCAdapter.cpp    # MCInstInfo implementation
├── CMakeLists.txt   # Build configuration
└── doc/
    └── Design.md    # This document
```
