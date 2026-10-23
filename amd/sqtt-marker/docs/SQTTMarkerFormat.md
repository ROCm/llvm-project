# SQTT marker format

Each marker header is a 32-bit shaderdata value. Bit 0 exits the previous
scope, bit 1 enters a scope, and bits 31:2 contain the marker ID. Thus a point
is `id << 2`, an entry is `(id << 2) | 2`, an exit is `1`, and an adjacent
exit/entry transition is `(id << 2) | 3`. IDs 1 through 63 can use
`s_ttracedata_imm` on gfx10 and later. Other values use `s_ttracedata` via M0,
with the target-required four-cycle M0 delay on gfx10 and later.

The pass writes a newline-delimited `.sqtt_funcmap` section:

- `F:id:name@location` identifies an instrumented function.
- `K:name@location` identifies a kernel; it has no marker ID.
- `U:id:name` identifies a user scope.
- `P:id:name[@location]` identifies a point or address-trace header.
- `R:id:extra_payload_count=N` declares following raw shaderdata records.
- `W:N` records the wave size for address payloads.
- `M:shader_clock_bits=N;shader_clock_shift=S` records gfx12 clock packing.

All feature classes share one ID space. Address-trace headers have unique IDs
and payload counts; ordinary rows implicitly have zero payloads. A decoder
must use the funcmap to distinguish headers from raw payload values.

When gfx12 shader-clock packing is enabled, the high `N` bits of pass-created
headers hold a clock sample beginning at source bit `S`. The remaining ID bits
retain the layout above. Payload-bearing markers require clock packing to be
disabled.

The pass runs before inlining to carry function and literal string markers
through optimization, then runs late to remove below-threshold functions,
compact IDs, add scope checks and ordering boundaries, instrument automatic
events, and emit the funcmap. At `-O0`, the late pass performs direct
instrumentation.

Address tracing emits a point header, EXEC low/high values, and then per-lane
payloads. Global/flat addresses are 64-bit; LDS and permute indices are 32-bit.
Raw buffer traces additionally record resource low/high and scalar offset.
The decoder uses the `R:` and `W:` rows to frame these records.
