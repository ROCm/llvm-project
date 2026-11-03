//===- ThinLinkFnAttrs.h - Function attributes of the thin link -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The function attributes that the ThinLTO thin link sets: the results of the
// thin-link passes, which every ThinLTO backend sets on its functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_THINLINKFNATTRS_H
#define LLVM_IR_THINLINKFNATTRS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <map>
#include <string>

namespace llvm {

class Function;

/// The function attributes that a thin-link pass sets on a function. Every
/// ThinLTO backend sets them on the function, also in a distributed build:
/// they are in a required summary extension, on each copy of the function, so
/// each backend gets them with the summaries that it receives.
class ThinLinkFnAttrs {
public:
  /// The name and the version of the summary extension. The entries hold the
  /// enum attributes by name, because the enum values change between versions
  /// of LLVM.
  static constexpr StringLiteral ExtensionName = "fn-attrs";
  static constexpr uint64_t ExtensionVersion = 1;

  /// Adds the enum attribute \p Kind.
  LLVM_ABI void add(Attribute::AttrKind Kind);

  /// Adds the string attribute \p Key with \p Value, or replaces its value.
  LLVM_ABI void add(StringRef Key, StringRef Value = "");

  /// Adds the attributes of \p Other. Its string values replace the values
  /// here.
  LLVM_ABI void add(const ThinLinkFnAttrs &Other);

  bool empty() const { return EnumAttrs.empty() && StringAttrs.empty(); }

  bool operator==(const ThinLinkFnAttrs &Other) const {
    return EnumAttrs == Other.EnumAttrs && StringAttrs == Other.StringAttrs;
  }

  /// Sets the attributes on \p F. A string attribute replaces the value that
  /// \p F has.
  LLVM_ABI void apply(Function &F) const;

  /// Returns the attributes of the summary \p S in \p Index. They are empty if
  /// no thin-link pass set attributes.
  LLVM_ABI static Expected<ThinLinkFnAttrs> get(const ModuleSummaryIndex &Index,
                                                const GlobalValueSummary &S);

  /// Adds \p Attrs to the attributes of each copy of \p VI in \p Index.
  LLVM_ABI static void addToSummaries(ModuleSummaryIndex &Index, ValueInfo VI,
                                      const ThinLinkFnAttrs &Attrs);

private:
  SummaryExtension::Entry toEntry() const;
  static Expected<ThinLinkFnAttrs> fromEntry(const SummaryExtension::Entry &E);

  // The enum attributes, sorted.
  SmallVector<Attribute::AttrKind, 4> EnumAttrs;
  // The string attributes, by key.
  std::map<std::string, std::string, std::less<>> StringAttrs;
};

} // end namespace llvm

#endif // LLVM_IR_THINLINKFNATTRS_H
