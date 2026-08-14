# AMD SQTT marker instrumentation

This project provides an LLVM pass plugin that inserts `s_ttracedata` markers
into AMDGPU code and a C-compatible device header for explicit user markers.
The trace decoding API remains in `rocprof-trace-decoder`.

The plugin is built and installed with LLVM by default. Disable it with
`-DLLVM_BUILD_SQTT_MARKER=OFF`. It can also be built and packaged separately:

```sh
cmake -S amd/sqtt-marker -B build-sqtt-marker \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build-sqtt-marker -j16
cmake --install build-sqtt-marker
cpack --config build-sqtt-marker/CPackConfig.cmake
```

Use the installed plugin and header with HIP:

```sh
SQTT_INSTRUMENT_FUNCTIONS=10 \
hipcc -DAMD_SQTT_MARKER_ENABLE=1 \
  -fpass-plugin=/path/to/lib/libsqtt-marker.so kernel.hip
```

```c
#include <amd_sqtt_marker/sqtt_marker.h>

amd_sqtt_marker_enter_string("work");
amd_sqtt_marker_data_string("item", item_id);
amd_sqtt_marker_exit_string("work");
```

String markers require the plugin. ID markers can be used without it:
`amd_sqtt_marker_enter_id`, `amd_sqtt_marker_exit`, and
`amd_sqtt_marker_point_id`. Marker calls are no-ops unless
`AMD_SQTT_MARKER_ENABLE` is nonzero.

Configuration is read by the compiler process. Supported variables are
`SQTT_INSTRUMENT_FUNCTIONS`, `SQTT_INSTRUMENT_BARRIERS`,
`SQTT_INSTRUMENT_MEMORY`, `SQTT_TRACE_ADDRESSES`, `SQTT_SCOPE_WAVE`,
`SQTT_SCOPE_SIMD`, `SQTT_SCOPE_CU`, `SQTT_SCOPE_WG`, `SQTT_MEM_BARRIER`,
`SQTT_SHADER_CLOCK_BITS`, and `SQTT_SHADER_CLOCK_SHIFT`. See
[the format note](docs/SQTTMarkerFormat.md) for the encoding and funcmap
contract.
