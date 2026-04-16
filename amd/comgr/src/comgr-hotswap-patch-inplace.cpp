//===- comgr-hotswap-patch-inplace.cpp - In-place B0-to-A0 patches --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Strong-symbol override for ApplyInPlacePatches.  Handles instruction
/// rewrites that fit in the same code size as the original:
///
///   - cluster_load -> global_load   (mnemonic swap, same encoding width)
///   - s_clause     -> s_nop         (4-byte byte-level overwrite)
///
/// No trampolines, ELF growth, or extra VGPRs are required.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

/// Map a B0-only cluster_load mnemonic to its A0-compatible global_load
/// equivalent.  Returns an empty StringRef if \p mnemonic is not a
/// cluster_load variant.
static llvm::StringRef GetClusterLoadReplacement(llvm::StringRef mnemonic) {
  return llvm::StringSwitch<llvm::StringRef>(mnemonic)
      .Case("cluster_load_b32", "global_load_b32")
      .Case("cluster_load_b64", "global_load_b64")
      .Case("cluster_load_b128", "global_load_b128")
      .Case("cluster_load_async_to_lds_b8", "global_load_async_to_lds_b8")
      .Case("cluster_load_async_to_lds_b32", "global_load_async_to_lds_b32")
      .Case("cluster_load_async_to_lds_b64", "global_load_async_to_lds_b64")
      .Case("cluster_load_async_to_lds_b128", "global_load_async_to_lds_b128")
      .Default("");
}

uint32_t ApplyInPlacePatches(PatchContext &ctx, size_t idx) {
  auto &di = ctx.decoded[idx];
  llvm::StringRef mnemonic(di.mnemonic);

  llvm::StringRef replacement = GetClusterLoadReplacement(mnemonic);
  if (!replacement.empty()) {
    RewriteRule rule;
    rule.replace_mnemonic = replacement.str();
    if (ApplyMnemonicSwap(rule, di, ctx.text, ctx.llvm_state)) {
      HotswapLog(HotswapLogLevel::Debug)
          << "hotswap: inplace: " << mnemonic << " -> " << replacement
          << " at 0x" << llvm::utohexstr(di.offset) << "\n";
      return 1;
    }
  }

  if (mnemonic == "s_clause") {
    RewriteRule rule;
    uint8_t nop[kMinInstSize];
    EncodeSNop(nop, ctx.config.s_nop_opcode);
    rule.replace_bytes.assign(nop, nop + kMinInstSize);
    if (ApplyByteReplace(rule, di.offset, di.size, ctx.text, ctx.text_size,
                         ctx.config.s_nop_opcode)) {
      HotswapLog(HotswapLogLevel::Debug)
          << "hotswap: inplace: s_clause -> s_nop at 0x"
          << llvm::utohexstr(di.offset) << "\n";
      return 1;
    }
  }

  return 0;
}
