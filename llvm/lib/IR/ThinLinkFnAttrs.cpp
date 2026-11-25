//===- ThinLinkFnAttrs.cpp - Function attributes from the thin link -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ThinLinkFnAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"

using namespace llvm;

void ThinLinkFnAttrs::add(Attribute::AttrKind Kind) {
  assert(Attribute::isEnumAttrKind(Kind) && Attribute::canUseAsFnAttr(Kind) &&
         "Expected an enum function attribute");
  auto It = llvm::lower_bound(EnumAttrs, Kind);
  if (It == EnumAttrs.end() || *It != Kind)
    EnumAttrs.insert(It, Kind);
}

void ThinLinkFnAttrs::add(StringRef Key, StringRef Value) {
  StringAttrs.insert_or_assign(Key.str(), Value.str());
}

void ThinLinkFnAttrs::add(const ThinLinkFnAttrs &Other) {
  for (Attribute::AttrKind Kind : Other.EnumAttrs)
    add(Kind);
  for (const auto &[Key, Value] : Other.StringAttrs)
    add(Key, Value);
}

void ThinLinkFnAttrs::apply(Function &F) const {
  for (Attribute::AttrKind Kind : EnumAttrs)
    F.addFnAttr(Kind);
  for (const auto &[Key, Value] : StringAttrs)
    F.addFnAttr(Key, Value);
}

// An entry has one integer: the number of enum attributes. Its strings are the
// names of the enum attributes, and then a key and a value for each string
// attribute.
SummaryExtension::Entry ThinLinkFnAttrs::toEntry() const {
  SummaryExtension::Entry E;
  E.Ints.push_back(EnumAttrs.size());
  for (Attribute::AttrKind Kind : EnumAttrs)
    E.Strings.push_back(Attribute::getNameFromAttrKind(Kind).str());
  for (const auto &[Key, Value] : StringAttrs) {
    E.Strings.push_back(Key);
    E.Strings.push_back(Value);
  }
  return E;
}

Expected<ThinLinkFnAttrs>
ThinLinkFnAttrs::fromEntry(const SummaryExtension::Entry &E) {
  if (E.Ints.size() != 1 || E.Ints[0] > E.Strings.size() ||
      (E.Strings.size() - E.Ints[0]) % 2 != 0)
    return createStringError(Twine("invalid entry in the summary extension '") +
                             ExtensionName + "'");
  ThinLinkFnAttrs Attrs;
  ArrayRef<std::string> Strings = E.Strings;
  for (const std::string &Name : Strings.take_front(E.Ints[0])) {
    Attribute::AttrKind Kind = Attribute::getAttrKindFromName(Name);
    if (!Attribute::isEnumAttrKind(Kind))
      return createStringError(Twine("unknown attribute '") + Name +
                               "' in the summary extension '" + ExtensionName +
                               "'");
    if (!Attribute::canUseAsFnAttr(Kind))
      return createStringError(Twine("attribute '") + Name +
                               "' in the summary extension '" + ExtensionName +
                               "' is not a function attribute");
    Attrs.add(Kind);
  }
  Strings = Strings.drop_front(E.Ints[0]);
  for (size_t I = 0; I != Strings.size(); I += 2)
    Attrs.add(Strings[I], Strings[I + 1]);
  return Attrs;
}

Expected<ThinLinkFnAttrs> ThinLinkFnAttrs::get(const ModuleSummaryIndex &Index,
                                               const GlobalValueSummary &S) {
  const SummaryExtension *Ext =
      Index.getExtension(ExtensionName, ExtensionVersion);
  if (!Ext)
    return ThinLinkFnAttrs();
  const SummaryExtension::Entry *E = Ext->getEntry(&S);
  if (!E)
    return ThinLinkFnAttrs();
  return fromEntry(*E);
}

void ThinLinkFnAttrs::addToSummaries(ModuleSummaryIndex &Index, ValueInfo VI,
                                     const ThinLinkFnAttrs &Attrs) {
  if (Attrs.empty())
    return;
  SummaryExtension &Ext = Index.getOrInsertExtension(
      ExtensionName, ExtensionVersion, /*Required=*/true);
  for (const auto &S : VI.getSummaryList()) {
    ThinLinkFnAttrs Merged;
    // An entry that is not valid (from a bad input file) is replaced.
    if (const SummaryExtension::Entry *E = Ext.getEntry(S.get())) {
      Expected<ThinLinkFnAttrs> Old = fromEntry(*E);
      if (Old)
        Merged = std::move(*Old);
      else
        consumeError(Old.takeError());
    }
    Merged.add(Attrs);
    Ext.setEntry(S.get(), Merged.toEntry());
  }
}
