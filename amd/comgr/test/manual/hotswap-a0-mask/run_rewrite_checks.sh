#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$ROOT/out}"
CLANG="${CLANG:-clang}"
HOTSWAP_REWRITE="${HOTSWAP_REWRITE:-hotswap-rewrite}"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-llvm-objdump}"
LLVM_READELF="${LLVM_READELF:-llvm-readelf}"
SOURCE_ISA="amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+"
TARGET_ISA="amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-"

mkdir -p "$OUT"

compile_kernel() {
  local src="$1"
  local dst="$2"
  "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib "$src" -o "$dst"
}

expect_grep() {
  local pattern="$1"
  local file="$2"
  if ! grep -Eq "$pattern" "$file"; then
    echo "missing pattern '$pattern' in $file" >&2
    return 1
  fi
}

echo "== compile B0 canary"
compile_kernel "$ROOT/b0_mask_rewrite_canary.s" \
  "$OUT/b0_mask_rewrite_canary.b0.hsaco"

echo "== rewrite B0 canary to A0"
AMD_COMGR_EMIT_VERBOSE_LOGS=1 "$HOTSWAP_REWRITE" \
  "$OUT/b0_mask_rewrite_canary.b0.hsaco" "$SOURCE_ISA" "$TARGET_ISA" \
  --output "$OUT/b0_mask_rewrite_canary.a0.hsaco" \
  >"$OUT/b0_mask_rewrite_canary.rewrite.log" 2>&1
expect_grep "RESULT: SUCCESS" "$OUT/b0_mask_rewrite_canary.rewrite.log"

"$LLVM_OBJDUMP" -d "$OUT/b0_mask_rewrite_canary.a0.hsaco" \
  >"$OUT/b0_mask_rewrite_canary.a0.disasm"
"$LLVM_READELF" --notes "$OUT/b0_mask_rewrite_canary.a0.hsaco" \
  >"$OUT/b0_mask_rewrite_canary.a0.notes"

echo "== verify patched instructions"
expect_grep "s_pack_hh_b32_b16 s4, 0, s4" \
  "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep "s_mov_b32 s[0-9]+, m0" \
  "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep "s_pack_hh_b32_b16 m0, 0, m0" \
  "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep "cluster_load_b64" "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep "cluster_load_async_to_lds_b32" \
  "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep "global_load_b32" "$OUT/b0_mask_rewrite_canary.a0.disasm"
expect_grep ".sgpr_count:[[:space:]]+27" \
  "$OUT/b0_mask_rewrite_canary.a0.notes"

echo "== verify tensor no-scratch failure"
compile_kernel "$ROOT/b0_tensor_no_scratch.s" \
  "$OUT/b0_tensor_no_scratch.b0.hsaco"
"$HOTSWAP_REWRITE" "$OUT/b0_tensor_no_scratch.b0.hsaco" \
  "$SOURCE_ISA" "$TARGET_ISA" --expect-status ERROR \
  >"$OUT/b0_tensor_no_scratch.rewrite.log" 2>&1
expect_grep "RESULT: ERROR" "$OUT/b0_tensor_no_scratch.rewrite.log"

echo "== verify cluster no-scratch failure"
compile_kernel "$ROOT/b0_cluster_no_scratch.s" \
  "$OUT/b0_cluster_no_scratch.b0.hsaco"
"$HOTSWAP_REWRITE" "$OUT/b0_cluster_no_scratch.b0.hsaco" \
  "$SOURCE_ISA" "$TARGET_ISA" --expect-status ERROR \
  >"$OUT/b0_cluster_no_scratch.rewrite.log" 2>&1
expect_grep "RESULT: ERROR" "$OUT/b0_cluster_no_scratch.rewrite.log"

if [[ -n "${RUNNER:-}" ]]; then
  echo "== dispatch rewritten canary"
  "$RUNNER" "$OUT/b0_mask_rewrite_canary.a0.hsaco" \
    b0_mask_rewrite_canary.kd
fi

echo "PASS: hotswap A0 mask smoke checks"
