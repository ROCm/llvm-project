# Comgr ISA metadata

`amd_comgr_get_isa_metadata()` describes a compilation target: its identity,
supported target features, and resource limits. This reference documents every
entry returned by that API. Use `amd_comgr_get_isa_count()` and
`amd_comgr_get_isa_name()` to enumerate the targets supported by the installed
Comgr library.

The values describe the target model shipped with Comgr. They do not query an
installed GPU or report the resources used by an individual kernel. A generic
processor name describes a compiler compatibility target that can cover more
than one GPU.

Comgr also exposes `amd_comgr_get_data_metadata()` for metadata embedded in a
code object, including kernel arguments and resource usage. That API follows
the code object's metadata format; see LLVM's [AMDHSA code object metadata
reference][code-object-metadata]. Its entries and versioning are separate from
the ISA metadata described here.

## Representation and access

The root is an `AMD_COMGR_METADATA_KIND_MAP` with 21 entries. Every value is an
`AMD_COMGR_METADATA_KIND_STRING` except `Features`, which is another map.
Resource counts are decimal strings, and Boolean flags are the strings `"0"`
and `"1"`. For example, a workgroup limit of 1024 is returned as `"1024"`.
Keys are case-sensitive; applications should look up keys without depending on
map iteration order.

Use `amd_comgr_metadata_lookup()` to obtain a value and
`amd_comgr_get_metadata_string()` to read a string. Querying the string length
with a null output buffer includes the terminating null character. Destroy the
root handle and handles returned by lookup with `amd_comgr_destroy_metadata()`.
The [public API header][api] documents the metadata traversal functions and
handle ownership rules.

## Target identity

These fields describe the requested ISA name. The name contains a target
triple, a processor component, and optional target feature settings. See
LLVM's descriptions of [target triples][target-triples] and
[AMDGPU target IDs][target-id].

| Entry | Description |
| --- | --- |
| `Name` | Complete ISA name passed to the query, including any feature settings. |
| `Architecture` | Architecture component of the requested name, such as `amdgcn` or an AMDGPU architecture with a subarchitecture. |
| `Vendor` | Vendor component of the requested name, normally `amd`. |
| `OS` | Operating-system or runtime-environment component of the target triple, such as `amdhsa`. |
| `Environment` | Environment component of the target triple. An omitted environment is returned as the empty string. |
| `Processor` | Processor component of the requested name, such as `gfx90a`. |
| `Version` | ISA metadata format version, currently `"1.0.0"`. This is separate from the GPU ISA revision, Comgr library version, and code object version. |

`Name` and its component fields preserve the spelling supplied to the query;
they are not a request to canonicalize the name.

## Target features

| Entry | Description |
| --- | --- |
| `Features` | Map of target feature settings supported by this ISA. The map can be empty. Its possible entries are `xnack` and `sramecc`, described below. |
| `Features.xnack` | Setting for XNACK memory-fault replay, which supports demand paging and page migration. |
| `Features.sramecc` | Setting for SRAM error-correcting code (ECC) support. |

Each feature value is a string:

| Value | Meaning |
| --- | --- |
| `"any"` | The ISA supports the feature setting, and the requested name leaves it unspecified. |
| `"on"` | The requested name explicitly includes `:xnack+` or `:sramecc+`. |
| `"off"` | The requested name explicitly includes `:xnack-` or `:sramecc-`. |

A feature without a supported setting is absent from the map. Requesting that
feature explicitly is an error. Absence describes Comgr's supported target
settings; it does not establish whether the hardware contains a particular
mechanism. The values express target requirements, not the current settings
of an installed GPU. LLVM's [target feature documentation][target-features]
explains their code-generation and code-object compatibility implications.

For example, the identity and feature portion of the metadata for
`amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-` is:

```yaml
Name: "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"
Architecture: "amdgcn"
Vendor: "amd"
OS: "amdhsa"
Environment: ""
Processor: "gfx90a"
Version: "1.0.0"
Features:
  sramecc: "on"
  xnack: "off"
```

## Capabilities

| Entry | Description |
| --- | --- |
| `TrapHandlerEnabled` | Whether Comgr models the target as having trap handling enabled. Currently `"1"` for every supported AMDGCN target. This flag does not inspect an installed trap handler or report its register reservation. |
| `ImageSupport` | Whether the target supports AMDGPU image instructions. Returned as `"1"` when LLVM's image-instruction feature is present, otherwise `"0"`. This is an ISA capability, not a complete image-format capability query for a runtime API. |

## Memory and execution limits

An execution unit (EU) here is a SIMD that executes wavefronts. A compute unit
(CU) and a workgroup processor (WGP) are different resource domains on RDNA:
a physical CU has two SIMDs, while a WGP has four. Comgr reports the full
four-SIMD block for `EUsPerCU` and `MaxWavesPerCU`.

| Entry | Unit | Description |
| --- | --- | --- |
| `LocalMemorySize` | Bytes per workgroup | Maximum LDS (local data share) memory that a single workgroup can address. Comgr queries `getMaxHWAddressableLocalMemorySize()`, the architectural per-workgroup cap. |
| `LDSBankCount` | Banks per CU | Number of LDS banks in LLVM's target model, as returned by `getLDSBankCount()`. Comgr forwards this count without CU/WGP scaling. |
| `EUsPerCU` | SIMDs per full block | Number of SIMDs in the full execution block, currently `"4"`. On RDNA this describes a WGP despite the historical `PerCU` name. Comgr queries `getNumWorkGroupSIMDs(true)`. |
| `MaxWavesPerCU` | Wavefronts per full block | Maximum resident-wave count before restrictions from register usage, LDS usage, or workgroup layout. Comgr multiplies `getMaxWavesPerEU()` by `EUsPerCU`; on RDNA the result is a WGP total. |
| `MaxFlatWorkGroupSize` | Work-items per workgroup | Maximum workgroup size supported by the compiler, currently `"1024"`. For a multidimensional workgroup this limits the product of its dimensions. Comgr queries `getMaxFlatWorkGroupSize()`. |

`LocalMemorySize` describes a different limit from the total physical LDS
available to all workgroups on a CU or WGP. For example, Comgr reports 32 KiB
for `gfx600` and 64 KiB for `gfx1030` through this per-workgroup query. These
values must not be used as the total physical capacity when calculating
LDS-limited occupancy. Likewise, `LDSBankCount` is not multiplied by the
four-SIMD factor used for the execution counts.

The execution counts use full-SIMD mode independently of a kernel's CU/WGP
execution mode. They are architectural upper bounds rather than a prediction
of a particular kernel's occupancy. The [TargetParser declarations][target-parser]
describe the underlying queries and their units.

## Scalar registers

An SGPR is a 32-bit scalar general-purpose register shared by the work-items
of a wavefront.

| Entry | Unit | Description |
| --- | --- | --- |
| `SGPRAllocGranule` | SGPRs per allocation unit | SGPR allocation granularity reported by `getSGPRAllocGranule()`. For targets where SGPRs limit occupancy, register use is allocated in multiples of this count. |
| `TotalNumSGPRs` | SGPRs per SIMD | Scalar-register capacity reported by `getTotalNumSGPRs()`, used by LLVM's SGPR occupancy model on targets where that limit applies. |
| `AddressableNumSGPRs` | SGPRs per wavefront | Maximum number of SGPRs a wavefront can address, as returned by `getAddressableNumSGPRs()`. This includes target-specific restrictions such as the SGPR initialization workaround. |

These are target-level counts. ABI requirements, special-register use, and
trap-handler reservations can further constrain a kernel's register budget.
They do not report the number of registers used by a compiled kernel.

On GFX10 and later, `SGPRAllocGranule` equals `AddressableNumSGPRs` in LLVM's
current model. LLVM does not treat SGPR usage as an occupancy limit on those
targets. Consequently, the two SGPR capacity fields alone do not define a
general occupancy formula. See the [TargetParser implementation][target-parser-impl]
and `isSGPROccupancyLimited()` in [AMDGPUBaseInfo.cpp][base-info].

## Vector registers

A VGPR is a vector general-purpose register with one 32-bit element per
wavefront lane. A count of VGPRs therefore differs from a count of individual
32-bit storage elements. LLVM's [register descriptions][registers] explain
this representation.

| Entry | Unit | Description |
| --- | --- | --- |
| `VGPRAllocGranule` | VGPRs per allocation unit | Number of VGPRs in an allocation unit for a wavefront, returned by `getVGPRAllocGranule()`. Register use is rounded to this granularity when computing resource occupancy. |
| `TotalNumVGPRs` | Wave-sized VGPRs per SIMD | Physical vector-register capacity shared by the waves on a SIMD, expressed in the reporting wave size described below. Returned by `getTotalNumVGPRs()`. |
| `AddressableNumVGPRs` | VGPRs per wavefront | Maximum number of VGPRs a single wavefront can address, returned by `getAddressableNumVGPRs()`. On targets that unify VGPRs and accumulator registers (AGPRs), the count includes that shared addressable capacity. |

Comgr uses **wave32 values on targets supporting wave32** and native **wave64
values on GFX6 through GFX9**. The implementation selects wave32 using LLVM's
`FEAT_GFX10_INSTS` feature when making all three VGPR queries. This is a
reporting convention: the queried ISA name does not select a kernel's wave
size.

For example, Comgr reports `TotalNumVGPRs = "1024"` and
`VGPRAllocGranule = "16"` for `gfx1030`. The corresponding LLVM wave64 queries
return 512 and 8. `AddressableNumVGPRs` is 256 in both modes for that target;
addressability follows its own target-specific rules, so it has no universal
factor-of-two conversion.

These queries describe static VGPR allocation. A kernel's dynamic VGPR block
size and resulting allocation limits are not parameters of the ISA metadata
API. Consult the [TargetParser declarations][target-parser] for the wave-size
parameters and dynamic-allocation exclusions.

## Implementation reference

[`getIsaMetadata()` in comgr-metadata.cpp][implementation] constructs this map.
Keep this reference and the schema comment above that function in sync when
adding fields or changing their meaning. The LLVM links describe the queries
and target conventions; the values returned by a particular Comgr release
follow the LLVM version built into that release.

[api]: ../include/amd_comgr.h.in
[implementation]: ../src/comgr-metadata.cpp
[code-object-metadata]: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-code-object-metadata
[target-triples]: https://llvm.org/docs/AMDGPUUsage.html#target-triples
[target-id]: https://llvm.org/docs/AMDGPUUsage.html#target-id
[target-features]: https://llvm.org/docs/AMDGPUUsage.html#target-features
[registers]: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-dwarf-register-mapping-table
[target-parser]: https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/TargetParser/AMDGPUTargetParser.h
[target-parser-impl]: https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/AMDGPUTargetParser.cpp
[base-info]: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
