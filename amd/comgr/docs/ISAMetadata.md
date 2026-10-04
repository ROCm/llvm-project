# Comgr ISA metadata

`amd_comgr_get_isa_metadata()` returns a target's identity, supported feature
settings, and resource limits. Values describe Comgr's compiler target model,
not an installed GPU or an individual kernel. A generic target can represent
multiple GPUs. [Code object metadata][code-object-metadata], returned by
`amd_comgr_get_data_metadata()`, has a separate schema.

## Representation and access

The root is an `AMD_COMGR_METADATA_KIND_MAP` with 21 entries. Values are
`AMD_COMGR_METADATA_KIND_STRING`, except `Features`, which is a map of strings.
Counts use decimal strings; Boolean flags use `"0"` and `"1"`. Keys are
case-sensitive, and map iteration order is unspecified.

Use `amd_comgr_metadata_lookup()` to obtain a value and
`amd_comgr_get_metadata_string()` to read a string. Destroy metadata handles
with `amd_comgr_destroy_metadata()`. See the [public API header][api] for access
and ownership details.

## Target identity

Identity fields preserve the spelling in the requested ISA name. See LLVM's
[target ID format][target-id].

| Entry | Description |
| --- | --- |
| `Name` | Complete requested ISA name, including feature settings. |
| `Architecture` | Architecture component, such as `amdgcn`. |
| `Vendor` | Vendor component of the requested name, normally `amd`. |
| `OS` | OS/runtime component, such as `amdhsa`. |
| `Environment` | Environment component; an empty string when omitted. |
| `Processor` | Processor name, such as `gfx90a`. |
| `Version` | ISA metadata schema version, currently `"1.0.0"`; separate from the GPU ISA, library, and code object versions. |

## Target features

| Entry | Description |
| --- | --- |
| `Features` | Map of supported target feature settings; may be empty. |
| `Features.xnack` | XNACK memory-fault replay setting. |
| `Features.sramecc` | SRAM error-correcting code (ECC) setting. |

Each feature value is a string describing a requirement of the requested ISA:

| Value | Meaning |
| --- | --- |
| `"any"` | Either on or off is permitted; the requested name leaves the setting unspecified. |
| `"on"` | Enabled is required for this target ID, selected by `:xnack+` or `:sramecc+`. |
| `"off"` | Disabled is required for this target ID, selected by `:xnack-` or `:sramecc-`. |

These requirements apply to the requested target ID, not every use of the
processor. A feature without a selectable setting is absent; requesting it
explicitly is an error. See LLVM's [target feature documentation][target-features]
for code-generation and compatibility rules.

## Capabilities

| Entry | Description |
| --- | --- |
| `TrapHandlerEnabled` | Whether Comgr models trap handling as enabled; currently `"1"` for every supported AMDGCN target. |
| `ImageSupport` | Support for AMDGPU image instructions: `"1"` when supported, otherwise `"0"`. |

## Memory and execution limits

An execution unit (EU) is a SIMD. On RDNA and CDNA5, a physical compute unit
(CU) has two SIMDs, while a workgroup processor (WGP) has four.

| Entry | Unit | Description |
| --- | --- | --- |
| `LocalMemorySize` | Bytes per full physical LDS block | Total shared LDS capacity in full-SIMD mode; per WGP on RDNA and CDNA5. This can exceed one workgroup's allocation limit. |
| `LDSBankCount` | Banks (target-dependent scope) | LDS bank count from LLVM's target model, without CU/WGP scaling. |
| `EUsPerCU` | SIMDs per physical CU | `"2"` on GFX10 and later, otherwise `"4"`. |
| `MaxWavesPerCU` | Wavefronts per physical CU | Maximum resident waves before kernel resource restrictions. |
| `MaxFlatWorkGroupSize` | Work-items per workgroup | Maximum product of the workgroup dimensions, currently `"1024"`. |

## Scalar registers

An SGPR is a 32-bit scalar general-purpose register shared by the work-items
of a wavefront.

| Entry | Unit | Description |
| --- | --- | --- |
| `SGPRAllocGranule` | SGPRs per allocation unit | Allocation granularity in LLVM's target model. |
| `TotalNumSGPRs` | SGPRs per SIMD | Capacity used by LLVM's occupancy model where SGPRs limit occupancy. |
| `AddressableNumSGPRs` | SGPRs per wavefront | Maximum addressable SGPRs, including target-specific restrictions. |

On GFX10 and later, `SGPRAllocGranule` and `AddressableNumSGPRs` are both
`"106"`, the fixed allowance of normal SGPRs per wavefront. `TotalNumSGPRs`
reports the legacy value `"800"`, which is not a physical capacity or an
occupancy limit on these targets.

## Vector registers

A VGPR is a vector general-purpose register with one 32-bit element per
wavefront lane. These fields describe static allocation using wave32 on GFX10
and later, and wave64 on GFX6 through GFX9.

| Entry | Unit | Description |
| --- | --- | --- |
| `VGPRAllocGranule` | VGPRs per allocation unit | Number of VGPRs allocated at a time to a wavefront. |
| `TotalNumVGPRs` | Wave-sized VGPRs per SIMD | Vector-register capacity shared by resident waves. |
| `AddressableNumVGPRs` | VGPRs per wavefront | Maximum addressable VGPRs; includes shared accumulator-register (AGPR) capacity on targets with a unified register file. |

## Implementation reference

[`getIsaMetadata()`][implementation] constructs the map using LLVM's
[TargetParser queries][target-parser]. Keep this reference and its source
schema comment in sync when changing fields.

[api]: ../include/amd_comgr.h.in
[implementation]: ../src/comgr-metadata.cpp
[code-object-metadata]: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-code-object-metadata
[target-id]: https://llvm.org/docs/AMDGPUUsage.html#target-id
[target-features]: https://llvm.org/docs/AMDGPUUsage.html#target-features
[target-parser]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/TargetParser/AMDGPUTargetParser.h
