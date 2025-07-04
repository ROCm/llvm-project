//===- llvm/Attributes.h - Container for Attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the simple types necessary to represent the
/// attributes associated with functions and their calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ATTRIBUTES_H
#define LLVM_IR_ATTRIBUTES_H

#include "llvm-c/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace llvm {

class AttrBuilder;
class AttributeMask;
class AttributeImpl;
class AttributeListImpl;
class AttributeSetNode;
class ConstantRange;
class ConstantRangeList;
class FoldingSetNodeID;
class Function;
class LLVMContext;
class Instruction;
class Type;
class raw_ostream;
enum FPClassTest : unsigned;

enum class AllocFnKind : uint64_t {
  Unknown = 0,
  Alloc = 1 << 0,         // Allocator function returns a new allocation
  Realloc = 1 << 1,       // Allocator function resizes the `allocptr` argument
  Free = 1 << 2,          // Allocator function frees the `allocptr` argument
  Uninitialized = 1 << 3, // Allocator function returns uninitialized memory
  Zeroed = 1 << 4,        // Allocator function returns zeroed memory
  Aligned = 1 << 5,       // Allocator function aligns allocations per the
                          // `allocalign` argument
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ Aligned)
};

//===----------------------------------------------------------------------===//
/// \class
/// Functions, function parameters, and return types can have attributes
/// to indicate how they should be treated by optimizations and code
/// generation. This class represents one of those attributes. It's light-weight
/// and should be passed around by-value.
class Attribute {
public:
  /// This enumeration lists the attributes that can be associated with
  /// parameters, function results, or the function itself.
  ///
  /// Note: The `uwtable' attribute is about the ABI or the user mandating an
  /// entry in the unwind table. The `nounwind' attribute is about an exception
  /// passing by the function.
  ///
  /// In a theoretical system that uses tables for profiling and SjLj for
  /// exceptions, they would be fully independent. In a normal system that uses
  /// tables for both, the semantics are:
  ///
  /// nil                = Needs an entry because an exception might pass by.
  /// nounwind           = No need for an entry
  /// uwtable            = Needs an entry because the ABI says so and because
  ///                      an exception might pass by.
  /// uwtable + nounwind = Needs an entry because the ABI says so.

  enum AttrKind {
    // IR-Level Attributes
    None,                  ///< No attributes have been set
    #define GET_ATTR_ENUM
    #include "llvm/IR/Attributes.inc"
    EndAttrKinds,          ///< Sentinel value useful for loops
    EmptyKey,              ///< Use as Empty key for DenseMap of AttrKind
    TombstoneKey,          ///< Use as Tombstone key for DenseMap of AttrKind
  };

  static const unsigned NumIntAttrKinds = LastIntAttr - FirstIntAttr + 1;
  static const unsigned NumTypeAttrKinds = LastTypeAttr - FirstTypeAttr + 1;

  static bool isEnumAttrKind(AttrKind Kind) {
    return Kind >= FirstEnumAttr && Kind <= LastEnumAttr;
  }
  static bool isIntAttrKind(AttrKind Kind) {
    return Kind >= FirstIntAttr && Kind <= LastIntAttr;
  }
  static bool isTypeAttrKind(AttrKind Kind) {
    return Kind >= FirstTypeAttr && Kind <= LastTypeAttr;
  }
  static bool isConstantRangeAttrKind(AttrKind Kind) {
    return Kind >= FirstConstantRangeAttr && Kind <= LastConstantRangeAttr;
  }
  static bool isConstantRangeListAttrKind(AttrKind Kind) {
    return Kind >= FirstConstantRangeListAttr &&
           Kind <= LastConstantRangeListAttr;
  }

  LLVM_ABI static bool canUseAsFnAttr(AttrKind Kind);
  LLVM_ABI static bool canUseAsParamAttr(AttrKind Kind);
  LLVM_ABI static bool canUseAsRetAttr(AttrKind Kind);

  LLVM_ABI static bool intersectMustPreserve(AttrKind Kind);
  LLVM_ABI static bool intersectWithAnd(AttrKind Kind);
  LLVM_ABI static bool intersectWithMin(AttrKind Kind);
  LLVM_ABI static bool intersectWithCustom(AttrKind Kind);

private:
  AttributeImpl *pImpl = nullptr;

  Attribute(AttributeImpl *A) : pImpl(A) {}

public:
  Attribute() = default;

  //===--------------------------------------------------------------------===//
  // Attribute Construction
  //===--------------------------------------------------------------------===//

  /// Return a uniquified Attribute object.
  LLVM_ABI static Attribute get(LLVMContext &Context, AttrKind Kind,
                                uint64_t Val = 0);
  LLVM_ABI static Attribute get(LLVMContext &Context, StringRef Kind,
                                StringRef Val = StringRef());
  LLVM_ABI static Attribute get(LLVMContext &Context, AttrKind Kind, Type *Ty);
  LLVM_ABI static Attribute get(LLVMContext &Context, AttrKind Kind,
                                const ConstantRange &CR);
  LLVM_ABI static Attribute get(LLVMContext &Context, AttrKind Kind,
                                ArrayRef<ConstantRange> Val);

  /// Return a uniquified Attribute object that has the specific
  /// alignment set.
  LLVM_ABI static Attribute getWithAlignment(LLVMContext &Context,
                                             Align Alignment);
  LLVM_ABI static Attribute getWithStackAlignment(LLVMContext &Context,
                                                  Align Alignment);
  LLVM_ABI static Attribute getWithDereferenceableBytes(LLVMContext &Context,
                                                        uint64_t Bytes);
  LLVM_ABI static Attribute
  getWithDereferenceableOrNullBytes(LLVMContext &Context, uint64_t Bytes);
  LLVM_ABI static Attribute
  getWithAllocSizeArgs(LLVMContext &Context, unsigned ElemSizeArg,
                       const std::optional<unsigned> &NumElemsArg);
  LLVM_ABI static Attribute getWithAllocKind(LLVMContext &Context,
                                             AllocFnKind Kind);
  LLVM_ABI static Attribute getWithVScaleRangeArgs(LLVMContext &Context,
                                                   unsigned MinValue,
                                                   unsigned MaxValue);
  LLVM_ABI static Attribute getWithByValType(LLVMContext &Context, Type *Ty);
  LLVM_ABI static Attribute getWithStructRetType(LLVMContext &Context,
                                                 Type *Ty);
  LLVM_ABI static Attribute getWithByRefType(LLVMContext &Context, Type *Ty);
  LLVM_ABI static Attribute getWithPreallocatedType(LLVMContext &Context,
                                                    Type *Ty);
  LLVM_ABI static Attribute getWithInAllocaType(LLVMContext &Context, Type *Ty);
  LLVM_ABI static Attribute getWithUWTableKind(LLVMContext &Context,
                                               UWTableKind Kind);
  LLVM_ABI static Attribute getWithMemoryEffects(LLVMContext &Context,
                                                 MemoryEffects ME);
  LLVM_ABI static Attribute getWithNoFPClass(LLVMContext &Context,
                                             FPClassTest Mask);
  LLVM_ABI static Attribute getWithCaptureInfo(LLVMContext &Context,
                                               CaptureInfo CI);

  /// For a typed attribute, return the equivalent attribute with the type
  /// changed to \p ReplacementTy.
  Attribute getWithNewType(LLVMContext &Context, Type *ReplacementTy) {
    assert(isTypeAttribute() && "this requires a typed attribute");
    return get(Context, getKindAsEnum(), ReplacementTy);
  }

  LLVM_ABI static Attribute::AttrKind getAttrKindFromName(StringRef AttrName);

  LLVM_ABI static StringRef getNameFromAttrKind(Attribute::AttrKind AttrKind);

  /// Return true if the provided string matches the IR name of an attribute.
  /// example: "noalias" return true but not "NoAlias"
  LLVM_ABI static bool isExistingAttribute(StringRef Name);

  //===--------------------------------------------------------------------===//
  // Attribute Accessors
  //===--------------------------------------------------------------------===//

  /// Return true if the attribute is an Attribute::AttrKind type.
  LLVM_ABI bool isEnumAttribute() const;

  /// Return true if the attribute is an integer attribute.
  LLVM_ABI bool isIntAttribute() const;

  /// Return true if the attribute is a string (target-dependent)
  /// attribute.
  LLVM_ABI bool isStringAttribute() const;

  /// Return true if the attribute is a type attribute.
  LLVM_ABI bool isTypeAttribute() const;

  /// Return true if the attribute is a ConstantRange attribute.
  LLVM_ABI bool isConstantRangeAttribute() const;

  /// Return true if the attribute is a ConstantRangeList attribute.
  LLVM_ABI bool isConstantRangeListAttribute() const;

  /// Return true if the attribute is any kind of attribute.
  bool isValid() const { return pImpl; }

  /// Return true if the attribute is present.
  LLVM_ABI bool hasAttribute(AttrKind Val) const;

  /// Return true if the target-dependent attribute is present.
  LLVM_ABI bool hasAttribute(StringRef Val) const;

  /// Returns true if the attribute's kind can be represented as an enum (Enum,
  /// Integer, Type, ConstantRange, or ConstantRangeList attribute).
  bool hasKindAsEnum() const { return !isStringAttribute(); }

  /// Return the attribute's kind as an enum (Attribute::AttrKind). This
  /// requires the attribute be representable as an enum (see: `hasKindAsEnum`).
  LLVM_ABI Attribute::AttrKind getKindAsEnum() const;

  /// Return the attribute's value as an integer. This requires that the
  /// attribute be an integer attribute.
  LLVM_ABI uint64_t getValueAsInt() const;

  /// Return the attribute's value as a boolean. This requires that the
  /// attribute be a string attribute.
  LLVM_ABI bool getValueAsBool() const;

  /// Return the attribute's kind as a string. This requires the
  /// attribute to be a string attribute.
  LLVM_ABI StringRef getKindAsString() const;

  /// Return the attribute's value as a string. This requires the
  /// attribute to be a string attribute.
  LLVM_ABI StringRef getValueAsString() const;

  /// Return the attribute's value as a Type. This requires the attribute to be
  /// a type attribute.
  LLVM_ABI Type *getValueAsType() const;

  /// Return the attribute's value as a ConstantRange. This requires the
  /// attribute to be a ConstantRange attribute.
  LLVM_ABI const ConstantRange &getValueAsConstantRange() const;

  /// Return the attribute's value as a ConstantRange array. This requires the
  /// attribute to be a ConstantRangeList attribute.
  LLVM_ABI ArrayRef<ConstantRange> getValueAsConstantRangeList() const;

  /// Returns the alignment field of an attribute as a byte alignment
  /// value.
  LLVM_ABI MaybeAlign getAlignment() const;

  /// Returns the stack alignment field of an attribute as a byte
  /// alignment value.
  LLVM_ABI MaybeAlign getStackAlignment() const;

  /// Returns the number of dereferenceable bytes from the
  /// dereferenceable attribute.
  LLVM_ABI uint64_t getDereferenceableBytes() const;

  /// Returns the number of dereferenceable_or_null bytes from the
  /// dereferenceable_or_null attribute.
  LLVM_ABI uint64_t getDereferenceableOrNullBytes() const;

  /// Returns the argument numbers for the allocsize attribute.
  LLVM_ABI std::pair<unsigned, std::optional<unsigned>>
  getAllocSizeArgs() const;

  /// Returns the minimum value for the vscale_range attribute.
  LLVM_ABI unsigned getVScaleRangeMin() const;

  /// Returns the maximum value for the vscale_range attribute or std::nullopt
  /// when unknown.
  LLVM_ABI std::optional<unsigned> getVScaleRangeMax() const;

  // Returns the unwind table kind.
  LLVM_ABI UWTableKind getUWTableKind() const;

  // Returns the allocator function kind.
  LLVM_ABI AllocFnKind getAllocKind() const;

  /// Returns memory effects.
  LLVM_ABI MemoryEffects getMemoryEffects() const;

  /// Returns information from captures attribute.
  LLVM_ABI CaptureInfo getCaptureInfo() const;

  /// Return the FPClassTest for nofpclass
  LLVM_ABI FPClassTest getNoFPClass() const;

  /// Return if global variable is instrumented by AddrSanitizer.
  bool isSanitizedPaddedGlobal() const;

  /// Returns the value of the range attribute.
  LLVM_ABI const ConstantRange &getRange() const;

  /// Returns the value of the initializes attribute.
  LLVM_ABI ArrayRef<ConstantRange> getInitializes() const;

  /// The Attribute is converted to a string of equivalent mnemonic. This
  /// is, presumably, for writing out the mnemonics for the assembly writer.
  LLVM_ABI std::string getAsString(bool InAttrGrp = false) const;

  /// Return true if this attribute belongs to the LLVMContext.
  LLVM_ABI bool hasParentContext(LLVMContext &C) const;

  /// Equality and non-equality operators.
  bool operator==(Attribute A) const { return pImpl == A.pImpl; }
  bool operator!=(Attribute A) const { return pImpl != A.pImpl; }

  /// Used to sort attribute by kind.
  LLVM_ABI int cmpKind(Attribute A) const;

  /// Less-than operator. Useful for sorting the attributes list.
  LLVM_ABI bool operator<(Attribute A) const;

  LLVM_ABI void Profile(FoldingSetNodeID &ID) const;

  /// Return a raw pointer that uniquely identifies this attribute.
  void *getRawPointer() const {
    return pImpl;
  }

  /// Get an attribute from a raw pointer created by getRawPointer.
  static Attribute fromRawPointer(void *RawPtr) {
    return Attribute(reinterpret_cast<AttributeImpl*>(RawPtr));
  }
};

// Specialized opaque value conversions.
inline LLVMAttributeRef wrap(Attribute Attr) {
  return reinterpret_cast<LLVMAttributeRef>(Attr.getRawPointer());
}

// Specialized opaque value conversions.
inline Attribute unwrap(LLVMAttributeRef Attr) {
  return Attribute::fromRawPointer(Attr);
}

//===----------------------------------------------------------------------===//
/// \class
/// This class holds the attributes for a particular argument, parameter,
/// function, or return value. It is an immutable value type that is cheap to
/// copy. Adding and removing enum attributes is intended to be fast, but adding
/// and removing string or integer attributes involves a FoldingSet lookup.
class AttributeSet {
  friend AttributeListImpl;
  template <typename Ty, typename Enable> friend struct DenseMapInfo;

  // TODO: Extract AvailableAttrs from AttributeSetNode and store them here.
  // This will allow an efficient implementation of addAttribute and
  // removeAttribute for enum attrs.

  /// Private implementation pointer.
  AttributeSetNode *SetNode = nullptr;

private:
  explicit AttributeSet(AttributeSetNode *ASN) : SetNode(ASN) {}

public:
  /// AttributeSet is a trivially copyable value type.
  AttributeSet() = default;
  AttributeSet(const AttributeSet &) = default;
  ~AttributeSet() = default;

  LLVM_ABI static AttributeSet get(LLVMContext &C, const AttrBuilder &B);
  LLVM_ABI static AttributeSet get(LLVMContext &C, ArrayRef<Attribute> Attrs);

  bool operator==(const AttributeSet &O) const { return SetNode == O.SetNode; }
  bool operator!=(const AttributeSet &O) const { return !(*this == O); }

  /// Add an argument attribute. Returns a new set because attribute sets are
  /// immutable.
  [[nodiscard]] LLVM_ABI AttributeSet
  addAttribute(LLVMContext &C, Attribute::AttrKind Kind) const;

  /// Add a target-dependent attribute. Returns a new set because attribute sets
  /// are immutable.
  [[nodiscard]] LLVM_ABI AttributeSet addAttribute(
      LLVMContext &C, StringRef Kind, StringRef Value = StringRef()) const;

  /// Add attributes to the attribute set. Returns a new set because attribute
  /// sets are immutable.
  [[nodiscard]] LLVM_ABI AttributeSet addAttributes(LLVMContext &C,
                                                    AttributeSet AS) const;

  /// Remove the specified attribute from this set. Returns a new set because
  /// attribute sets are immutable.
  [[nodiscard]] LLVM_ABI AttributeSet
  removeAttribute(LLVMContext &C, Attribute::AttrKind Kind) const;

  /// Remove the specified attribute from this set. Returns a new set because
  /// attribute sets are immutable.
  [[nodiscard]] LLVM_ABI AttributeSet removeAttribute(LLVMContext &C,
                                                      StringRef Kind) const;

  /// Remove the specified attributes from this set. Returns a new set because
  /// attribute sets are immutable.
  [[nodiscard]] LLVM_ABI AttributeSet
  removeAttributes(LLVMContext &C, const AttributeMask &AttrsToRemove) const;

  /// Try to intersect this AttributeSet with Other. Returns std::nullopt if
  /// the two lists are inherently incompatible (imply different behavior, not
  /// just analysis).
  [[nodiscard]] LLVM_ABI std::optional<AttributeSet>
  intersectWith(LLVMContext &C, AttributeSet Other) const;

  /// Return the number of attributes in this set.
  LLVM_ABI unsigned getNumAttributes() const;

  /// Return true if attributes exists in this set.
  bool hasAttributes() const { return SetNode != nullptr; }

  /// Return true if the attribute exists in this set.
  LLVM_ABI bool hasAttribute(Attribute::AttrKind Kind) const;

  /// Return true if the attribute exists in this set.
  LLVM_ABI bool hasAttribute(StringRef Kind) const;

  /// Return the attribute object.
  LLVM_ABI Attribute getAttribute(Attribute::AttrKind Kind) const;

  /// Return the target-dependent attribute object.
  LLVM_ABI Attribute getAttribute(StringRef Kind) const;

  LLVM_ABI MaybeAlign getAlignment() const;
  LLVM_ABI MaybeAlign getStackAlignment() const;
  LLVM_ABI uint64_t getDereferenceableBytes() const;
  LLVM_ABI uint64_t getDereferenceableOrNullBytes() const;
  LLVM_ABI Type *getByValType() const;
  LLVM_ABI Type *getStructRetType() const;
  LLVM_ABI Type *getByRefType() const;
  LLVM_ABI Type *getPreallocatedType() const;
  LLVM_ABI Type *getInAllocaType() const;
  LLVM_ABI Type *getElementType() const;
  LLVM_ABI std::optional<std::pair<unsigned, std::optional<unsigned>>>
  getAllocSizeArgs() const;
  LLVM_ABI unsigned getVScaleRangeMin() const;
  LLVM_ABI std::optional<unsigned> getVScaleRangeMax() const;
  LLVM_ABI UWTableKind getUWTableKind() const;
  LLVM_ABI AllocFnKind getAllocKind() const;
  LLVM_ABI MemoryEffects getMemoryEffects() const;
  LLVM_ABI CaptureInfo getCaptureInfo() const;
  LLVM_ABI FPClassTest getNoFPClass() const;
  LLVM_ABI std::string getAsString(bool InAttrGrp = false) const;

  /// Return true if this attribute set belongs to the LLVMContext.
  LLVM_ABI bool hasParentContext(LLVMContext &C) const;

  using iterator = const Attribute *;

  LLVM_ABI iterator begin() const;
  LLVM_ABI iterator end() const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const;
#endif
};

//===----------------------------------------------------------------------===//
/// \class
/// Provide DenseMapInfo for AttributeSet.
template <> struct DenseMapInfo<AttributeSet, void> {
  static AttributeSet getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
    return AttributeSet(reinterpret_cast<AttributeSetNode *>(Val));
  }

  static AttributeSet getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(-2);
    Val <<= PointerLikeTypeTraits<void *>::NumLowBitsAvailable;
    return AttributeSet(reinterpret_cast<AttributeSetNode *>(Val));
  }

  static unsigned getHashValue(AttributeSet AS) {
    return (unsigned((uintptr_t)AS.SetNode) >> 4) ^
           (unsigned((uintptr_t)AS.SetNode) >> 9);
  }

  static bool isEqual(AttributeSet LHS, AttributeSet RHS) { return LHS == RHS; }
};

//===----------------------------------------------------------------------===//
/// \class
/// This class holds the attributes for a function, its return value, and
/// its parameters. You access the attributes for each of them via an index into
/// the AttributeList object. The function attributes are at index
/// `AttributeList::FunctionIndex', the return value is at index
/// `AttributeList::ReturnIndex', and the attributes for the parameters start at
/// index `AttributeList::FirstArgIndex'.
class AttributeList {
public:
  enum AttrIndex : unsigned {
    ReturnIndex = 0U,
    FunctionIndex = ~0U,
    FirstArgIndex = 1,
  };

private:
  friend class AttrBuilder;
  friend class AttributeListImpl;
  friend class AttributeSet;
  friend class AttributeSetNode;
  template <typename Ty, typename Enable> friend struct DenseMapInfo;

  /// The attributes that we are managing. This can be null to represent
  /// the empty attributes list.
  AttributeListImpl *pImpl = nullptr;

public:
  /// Create an AttributeList with the specified parameters in it.
  LLVM_ABI static AttributeList
  get(LLVMContext &C, ArrayRef<std::pair<unsigned, Attribute>> Attrs);
  LLVM_ABI static AttributeList
  get(LLVMContext &C, ArrayRef<std::pair<unsigned, AttributeSet>> Attrs);

  /// Create an AttributeList from attribute sets for a function, its
  /// return value, and all of its arguments.
  LLVM_ABI static AttributeList get(LLVMContext &C, AttributeSet FnAttrs,
                                    AttributeSet RetAttrs,
                                    ArrayRef<AttributeSet> ArgAttrs);

private:
  explicit AttributeList(AttributeListImpl *LI) : pImpl(LI) {}

  static AttributeList getImpl(LLVMContext &C, ArrayRef<AttributeSet> AttrSets);

  AttributeList setAttributesAtIndex(LLVMContext &C, unsigned Index,
                                     AttributeSet Attrs) const;

public:
  AttributeList() = default;

  //===--------------------------------------------------------------------===//
  // AttributeList Construction and Mutation
  //===--------------------------------------------------------------------===//

  /// Return an AttributeList with the specified parameters in it.
  LLVM_ABI static AttributeList get(LLVMContext &C,
                                    ArrayRef<AttributeList> Attrs);
  LLVM_ABI static AttributeList get(LLVMContext &C, unsigned Index,
                                    ArrayRef<Attribute::AttrKind> Kinds);
  LLVM_ABI static AttributeList get(LLVMContext &C, unsigned Index,
                                    ArrayRef<Attribute::AttrKind> Kinds,
                                    ArrayRef<uint64_t> Values);
  LLVM_ABI static AttributeList get(LLVMContext &C, unsigned Index,
                                    ArrayRef<StringRef> Kind);
  LLVM_ABI static AttributeList get(LLVMContext &C, unsigned Index,
                                    AttributeSet Attrs);
  LLVM_ABI static AttributeList get(LLVMContext &C, unsigned Index,
                                    const AttrBuilder &B);

  // TODO: remove non-AtIndex versions of these methods.
  /// Add an attribute to the attribute set at the given index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList addAttributeAtIndex(
      LLVMContext &C, unsigned Index, Attribute::AttrKind Kind) const;

  /// Add an attribute to the attribute set at the given index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  addAttributeAtIndex(LLVMContext &C, unsigned Index, StringRef Kind,
                      StringRef Value = StringRef()) const;

  /// Add an attribute to the attribute set at the given index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList addAttributeAtIndex(LLVMContext &C,
                                                           unsigned Index,
                                                           Attribute A) const;

  /// Add attributes to the attribute set at the given index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList addAttributesAtIndex(
      LLVMContext &C, unsigned Index, const AttrBuilder &B) const;

  /// Add a function attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addFnAttribute(LLVMContext &C,
                                             Attribute::AttrKind Kind) const {
    return addAttributeAtIndex(C, FunctionIndex, Kind);
  }

  /// Add a function attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addFnAttribute(LLVMContext &C,
                                             Attribute Attr) const {
    return addAttributeAtIndex(C, FunctionIndex, Attr);
  }

  /// Add a function attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList
  addFnAttribute(LLVMContext &C, StringRef Kind,
                 StringRef Value = StringRef()) const {
    return addAttributeAtIndex(C, FunctionIndex, Kind, Value);
  }

  /// Add function attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addFnAttributes(LLVMContext &C,
                                              const AttrBuilder &B) const {
    return addAttributesAtIndex(C, FunctionIndex, B);
  }

  /// Add a return value attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addRetAttribute(LLVMContext &C,
                                              Attribute::AttrKind Kind) const {
    return addAttributeAtIndex(C, ReturnIndex, Kind);
  }

  /// Add a return value attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addRetAttribute(LLVMContext &C,
                                              Attribute Attr) const {
    return addAttributeAtIndex(C, ReturnIndex, Attr);
  }

  /// Add a return value attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addRetAttributes(LLVMContext &C,
                                               const AttrBuilder &B) const {
    return addAttributesAtIndex(C, ReturnIndex, B);
  }

  /// Add an argument attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList
  addParamAttribute(LLVMContext &C, unsigned ArgNo,
                    Attribute::AttrKind Kind) const {
    return addAttributeAtIndex(C, ArgNo + FirstArgIndex, Kind);
  }

  /// Add an argument attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList
  addParamAttribute(LLVMContext &C, unsigned ArgNo, StringRef Kind,
                    StringRef Value = StringRef()) const {
    return addAttributeAtIndex(C, ArgNo + FirstArgIndex, Kind, Value);
  }

  /// Add an attribute to the attribute list at the given arg indices. Returns a
  /// new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList addParamAttribute(
      LLVMContext &C, ArrayRef<unsigned> ArgNos, Attribute A) const;

  /// Add an argument attribute to the list. Returns a new list because
  /// attribute lists are immutable.
  [[nodiscard]] AttributeList addParamAttributes(LLVMContext &C, unsigned ArgNo,
                                                 const AttrBuilder &B) const {
    return addAttributesAtIndex(C, ArgNo + FirstArgIndex, B);
  }

  /// Remove the specified attribute at the specified index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList removeAttributeAtIndex(
      LLVMContext &C, unsigned Index, Attribute::AttrKind Kind) const;

  /// Remove the specified attribute at the specified index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  removeAttributeAtIndex(LLVMContext &C, unsigned Index, StringRef Kind) const;
  [[nodiscard]] AttributeList removeAttribute(LLVMContext &C, unsigned Index,
                                              StringRef Kind) const {
    return removeAttributeAtIndex(C, Index, Kind);
  }

  /// Remove the specified attributes at the specified index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList removeAttributesAtIndex(
      LLVMContext &C, unsigned Index, const AttributeMask &AttrsToRemove) const;

  /// Remove all attributes at the specified index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  removeAttributesAtIndex(LLVMContext &C, unsigned Index) const;

  /// Remove the specified attribute at the function index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeFnAttribute(LLVMContext &C, Attribute::AttrKind Kind) const {
    return removeAttributeAtIndex(C, FunctionIndex, Kind);
  }

  /// Remove the specified attribute at the function index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList removeFnAttribute(LLVMContext &C,
                                                StringRef Kind) const {
    return removeAttributeAtIndex(C, FunctionIndex, Kind);
  }

  /// Remove the specified attribute at the function index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeFnAttributes(LLVMContext &C, const AttributeMask &AttrsToRemove) const {
    return removeAttributesAtIndex(C, FunctionIndex, AttrsToRemove);
  }

  /// Remove the attributes at the function index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList removeFnAttributes(LLVMContext &C) const {
    return removeAttributesAtIndex(C, FunctionIndex);
  }

  /// Remove the specified attribute at the return value index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeRetAttribute(LLVMContext &C, Attribute::AttrKind Kind) const {
    return removeAttributeAtIndex(C, ReturnIndex, Kind);
  }

  /// Remove the specified attribute at the return value index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList removeRetAttribute(LLVMContext &C,
                                                 StringRef Kind) const {
    return removeAttributeAtIndex(C, ReturnIndex, Kind);
  }

  /// Remove the specified attribute at the return value index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeRetAttributes(LLVMContext &C,
                      const AttributeMask &AttrsToRemove) const {
    return removeAttributesAtIndex(C, ReturnIndex, AttrsToRemove);
  }

  /// Remove the specified attribute at the specified arg index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeParamAttribute(LLVMContext &C, unsigned ArgNo,
                       Attribute::AttrKind Kind) const {
    return removeAttributeAtIndex(C, ArgNo + FirstArgIndex, Kind);
  }

  /// Remove the specified attribute at the specified arg index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeParamAttribute(LLVMContext &C, unsigned ArgNo, StringRef Kind) const {
    return removeAttributeAtIndex(C, ArgNo + FirstArgIndex, Kind);
  }

  /// Remove the specified attribute at the specified arg index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList
  removeParamAttributes(LLVMContext &C, unsigned ArgNo,
                        const AttributeMask &AttrsToRemove) const {
    return removeAttributesAtIndex(C, ArgNo + FirstArgIndex, AttrsToRemove);
  }

  /// Remove all attributes at the specified arg index from this
  /// attribute list. Returns a new list because attribute lists are immutable.
  [[nodiscard]] AttributeList removeParamAttributes(LLVMContext &C,
                                                    unsigned ArgNo) const {
    return removeAttributesAtIndex(C, ArgNo + FirstArgIndex);
  }

  /// Replace the type contained by attribute \p AttrKind at index \p ArgNo wih
  /// \p ReplacementTy, preserving all other attributes.
  [[nodiscard]] AttributeList
  replaceAttributeTypeAtIndex(LLVMContext &C, unsigned ArgNo,
                              Attribute::AttrKind Kind,
                              Type *ReplacementTy) const {
    Attribute Attr = getAttributeAtIndex(ArgNo, Kind);
    auto Attrs = removeAttributeAtIndex(C, ArgNo, Kind);
    return Attrs.addAttributeAtIndex(C, ArgNo,
                                     Attr.getWithNewType(C, ReplacementTy));
  }

  /// \brief Add the dereferenceable attribute to the attribute set at the given
  /// index. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  addDereferenceableRetAttr(LLVMContext &C, uint64_t Bytes) const;

  /// \brief Add the dereferenceable attribute to the attribute set at the given
  /// arg index. Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList addDereferenceableParamAttr(
      LLVMContext &C, unsigned ArgNo, uint64_t Bytes) const;

  /// Add the dereferenceable_or_null attribute to the attribute set at
  /// the given arg index. Returns a new list because attribute lists are
  /// immutable.
  [[nodiscard]] LLVM_ABI AttributeList addDereferenceableOrNullParamAttr(
      LLVMContext &C, unsigned ArgNo, uint64_t Bytes) const;

  /// Add the range attribute to the attribute set at the return value index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  addRangeRetAttr(LLVMContext &C, const ConstantRange &CR) const;

  /// Add the allocsize attribute to the attribute set at the given arg index.
  /// Returns a new list because attribute lists are immutable.
  [[nodiscard]] LLVM_ABI AttributeList
  addAllocSizeParamAttr(LLVMContext &C, unsigned ArgNo, unsigned ElemSizeArg,
                        const std::optional<unsigned> &NumElemsArg) const;

  /// Try to intersect this AttributeList with Other. Returns std::nullopt if
  /// the two lists are inherently incompatible (imply different behavior, not
  /// just analysis).
  [[nodiscard]] LLVM_ABI std::optional<AttributeList>
  intersectWith(LLVMContext &C, AttributeList Other) const;

  //===--------------------------------------------------------------------===//
  // AttributeList Accessors
  //===--------------------------------------------------------------------===//

  /// The attributes for the specified index are returned.
  LLVM_ABI AttributeSet getAttributes(unsigned Index) const;

  /// The attributes for the argument or parameter at the given index are
  /// returned.
  LLVM_ABI AttributeSet getParamAttrs(unsigned ArgNo) const;

  /// The attributes for the ret value are returned.
  LLVM_ABI AttributeSet getRetAttrs() const;

  /// The function attributes are returned.
  LLVM_ABI AttributeSet getFnAttrs() const;

  /// Return true if the attribute exists at the given index.
  LLVM_ABI bool hasAttributeAtIndex(unsigned Index,
                                    Attribute::AttrKind Kind) const;

  /// Return true if the attribute exists at the given index.
  LLVM_ABI bool hasAttributeAtIndex(unsigned Index, StringRef Kind) const;

  /// Return true if attribute exists at the given index.
  LLVM_ABI bool hasAttributesAtIndex(unsigned Index) const;

  /// Return true if the attribute exists for the given argument
  bool hasParamAttr(unsigned ArgNo, Attribute::AttrKind Kind) const {
    return hasAttributeAtIndex(ArgNo + FirstArgIndex, Kind);
  }

  /// Return true if the attribute exists for the given argument
  bool hasParamAttr(unsigned ArgNo, StringRef Kind) const {
    return hasAttributeAtIndex(ArgNo + FirstArgIndex, Kind);
  }

  /// Return true if attributes exists for the given argument
  bool hasParamAttrs(unsigned ArgNo) const {
    return hasAttributesAtIndex(ArgNo + FirstArgIndex);
  }

  /// Return true if the attribute exists for the return value.
  bool hasRetAttr(Attribute::AttrKind Kind) const {
    return hasAttributeAtIndex(ReturnIndex, Kind);
  }

  /// Return true if the attribute exists for the return value.
  bool hasRetAttr(StringRef Kind) const {
    return hasAttributeAtIndex(ReturnIndex, Kind);
  }

  /// Return true if attributes exist for the return value.
  bool hasRetAttrs() const { return hasAttributesAtIndex(ReturnIndex); }

  /// Return true if the attribute exists for the function.
  LLVM_ABI bool hasFnAttr(Attribute::AttrKind Kind) const;

  /// Return true if the attribute exists for the function.
  LLVM_ABI bool hasFnAttr(StringRef Kind) const;

  /// Return true the attributes exist for the function.
  bool hasFnAttrs() const { return hasAttributesAtIndex(FunctionIndex); }

  /// Return true if the specified attribute is set for at least one
  /// parameter or for the return value. If Index is not nullptr, the index
  /// of a parameter with the specified attribute is provided.
  LLVM_ABI bool hasAttrSomewhere(Attribute::AttrKind Kind,
                                 unsigned *Index = nullptr) const;

  /// Return the attribute object that exists at the given index.
  LLVM_ABI Attribute getAttributeAtIndex(unsigned Index,
                                         Attribute::AttrKind Kind) const;

  /// Return the attribute object that exists at the given index.
  LLVM_ABI Attribute getAttributeAtIndex(unsigned Index, StringRef Kind) const;

  /// Return the attribute object that exists at the arg index.
  Attribute getParamAttr(unsigned ArgNo, Attribute::AttrKind Kind) const {
    return getAttributeAtIndex(ArgNo + FirstArgIndex, Kind);
  }

  /// Return the attribute object that exists at the given index.
  Attribute getParamAttr(unsigned ArgNo, StringRef Kind) const {
    return getAttributeAtIndex(ArgNo + FirstArgIndex, Kind);
  }

  /// Return the attribute object that exists for the function.
  Attribute getFnAttr(Attribute::AttrKind Kind) const {
    return getAttributeAtIndex(FunctionIndex, Kind);
  }

  /// Return the attribute object that exists for the function.
  Attribute getFnAttr(StringRef Kind) const {
    return getAttributeAtIndex(FunctionIndex, Kind);
  }

  /// Return the attribute for the given attribute kind for the return value.
  Attribute getRetAttr(Attribute::AttrKind Kind) const {
    return getAttributeAtIndex(ReturnIndex, Kind);
  }

  /// Return the alignment of the return value.
  LLVM_ABI MaybeAlign getRetAlignment() const;

  /// Return the alignment for the specified function parameter.
  LLVM_ABI MaybeAlign getParamAlignment(unsigned ArgNo) const;

  /// Return the stack alignment for the specified function parameter.
  LLVM_ABI MaybeAlign getParamStackAlignment(unsigned ArgNo) const;

  /// Return the byval type for the specified function parameter.
  LLVM_ABI Type *getParamByValType(unsigned ArgNo) const;

  /// Return the sret type for the specified function parameter.
  LLVM_ABI Type *getParamStructRetType(unsigned ArgNo) const;

  /// Return the byref type for the specified function parameter.
  LLVM_ABI Type *getParamByRefType(unsigned ArgNo) const;

  /// Return the preallocated type for the specified function parameter.
  LLVM_ABI Type *getParamPreallocatedType(unsigned ArgNo) const;

  /// Return the inalloca type for the specified function parameter.
  LLVM_ABI Type *getParamInAllocaType(unsigned ArgNo) const;

  /// Return the elementtype type for the specified function parameter.
  LLVM_ABI Type *getParamElementType(unsigned ArgNo) const;

  /// Get the stack alignment of the function.
  LLVM_ABI MaybeAlign getFnStackAlignment() const;

  /// Get the stack alignment of the return value.
  LLVM_ABI MaybeAlign getRetStackAlignment() const;

  /// Get the number of dereferenceable bytes (or zero if unknown) of the return
  /// value.
  LLVM_ABI uint64_t getRetDereferenceableBytes() const;

  /// Get the number of dereferenceable bytes (or zero if unknown) of an arg.
  LLVM_ABI uint64_t getParamDereferenceableBytes(unsigned Index) const;

  /// Get the number of dereferenceable_or_null bytes (or zero if unknown) of
  /// the return value.
  LLVM_ABI uint64_t getRetDereferenceableOrNullBytes() const;

  /// Get the number of dereferenceable_or_null bytes (or zero if unknown) of an
  /// arg.
  LLVM_ABI uint64_t getParamDereferenceableOrNullBytes(unsigned ArgNo) const;

  /// Get range (or std::nullopt if unknown) of an arg.
  LLVM_ABI std::optional<ConstantRange> getParamRange(unsigned ArgNo) const;

  /// Get the disallowed floating-point classes of the return value.
  LLVM_ABI FPClassTest getRetNoFPClass() const;

  /// Get the disallowed floating-point classes of the argument value.
  LLVM_ABI FPClassTest getParamNoFPClass(unsigned ArgNo) const;

  /// Get the unwind table kind requested for the function.
  LLVM_ABI UWTableKind getUWTableKind() const;

  LLVM_ABI AllocFnKind getAllocKind() const;

  /// Returns memory effects of the function.
  LLVM_ABI MemoryEffects getMemoryEffects() const;

  /// Return the attributes at the index as a string.
  LLVM_ABI std::string getAsString(unsigned Index,
                                   bool InAttrGrp = false) const;

  /// Return true if this attribute list belongs to the LLVMContext.
  LLVM_ABI bool hasParentContext(LLVMContext &C) const;

  //===--------------------------------------------------------------------===//
  // AttributeList Introspection
  //===--------------------------------------------------------------------===//

  using iterator = const AttributeSet *;

  LLVM_ABI iterator begin() const;
  LLVM_ABI iterator end() const;

  LLVM_ABI unsigned getNumAttrSets() const;

  // Implementation of indexes(). Produces iterators that wrap an index. Mostly
  // to hide the awkwardness of unsigned wrapping when iterating over valid
  // indexes.
  struct index_iterator {
    unsigned NumAttrSets;
    index_iterator(int NumAttrSets) : NumAttrSets(NumAttrSets) {}
    struct int_wrapper {
      int_wrapper(unsigned i) : i(i) {}
      unsigned i;
      unsigned operator*() { return i; }
      bool operator!=(const int_wrapper &Other) { return i != Other.i; }
      int_wrapper &operator++() {
        // This is expected to undergo unsigned wrapping since FunctionIndex is
        // ~0 and that's where we start.
        ++i;
        return *this;
      }
    };

    int_wrapper begin() { return int_wrapper(AttributeList::FunctionIndex); }

    int_wrapper end() { return int_wrapper(NumAttrSets - 1); }
  };

  /// Use this to iterate over the valid attribute indexes.
  index_iterator indexes() const { return index_iterator(getNumAttrSets()); }

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttributeList &RHS) const { return pImpl == RHS.pImpl; }
  bool operator!=(const AttributeList &RHS) const { return pImpl != RHS.pImpl; }

  /// Return a raw pointer that uniquely identifies this attribute list.
  void *getRawPointer() const {
    return pImpl;
  }

  /// Return true if there are no attributes.
  bool isEmpty() const { return pImpl == nullptr; }

  LLVM_ABI void print(raw_ostream &O) const;

  LLVM_ABI void dump() const;
};

//===----------------------------------------------------------------------===//
/// \class
/// Provide DenseMapInfo for AttributeList.
template <> struct DenseMapInfo<AttributeList, void> {
  static AttributeList getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<void*>::NumLowBitsAvailable;
    return AttributeList(reinterpret_cast<AttributeListImpl *>(Val));
  }

  static AttributeList getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(-2);
    Val <<= PointerLikeTypeTraits<void*>::NumLowBitsAvailable;
    return AttributeList(reinterpret_cast<AttributeListImpl *>(Val));
  }

  static unsigned getHashValue(AttributeList AS) {
    return (unsigned((uintptr_t)AS.pImpl) >> 4) ^
           (unsigned((uintptr_t)AS.pImpl) >> 9);
  }

  static bool isEqual(AttributeList LHS, AttributeList RHS) {
    return LHS == RHS;
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// This class is used in conjunction with the Attribute::get method to
/// create an Attribute object. The object itself is uniquified. The Builder's
/// value, however, is not. So this can be used as a quick way to test for
/// equality, presence of attributes, etc.
class AttrBuilder {
  LLVMContext &Ctx;
  SmallVector<Attribute, 8> Attrs;

public:
  AttrBuilder(LLVMContext &Ctx) : Ctx(Ctx) {}
  AttrBuilder(const AttrBuilder &) = delete;
  AttrBuilder(AttrBuilder &&) = default;

  AttrBuilder(LLVMContext &Ctx, const Attribute &A) : Ctx(Ctx) {
    addAttribute(A);
  }

  LLVM_ABI AttrBuilder(LLVMContext &Ctx, AttributeSet AS);

  LLVM_ABI void clear();

  /// Add an attribute to the builder.
  LLVM_ABI AttrBuilder &addAttribute(Attribute::AttrKind Val);

  /// Add the Attribute object to the builder.
  LLVM_ABI AttrBuilder &addAttribute(Attribute A);

  /// Add the target-dependent attribute to the builder.
  LLVM_ABI AttrBuilder &addAttribute(StringRef A, StringRef V = StringRef());

  /// Remove an attribute from the builder.
  LLVM_ABI AttrBuilder &removeAttribute(Attribute::AttrKind Val);

  /// Remove the target-dependent attribute from the builder.
  LLVM_ABI AttrBuilder &removeAttribute(StringRef A);

  /// Remove the target-dependent attribute from the builder.
  AttrBuilder &removeAttribute(Attribute A) {
    if (A.isStringAttribute())
      return removeAttribute(A.getKindAsString());
    else
      return removeAttribute(A.getKindAsEnum());
  }

  /// Add the attributes from the builder. Attributes in the passed builder
  /// overwrite attributes in this builder if they have the same key.
  LLVM_ABI AttrBuilder &merge(const AttrBuilder &B);

  /// Remove the attributes from the builder.
  LLVM_ABI AttrBuilder &remove(const AttributeMask &AM);

  /// Return true if the builder has any attribute that's in the
  /// specified builder.
  LLVM_ABI bool overlaps(const AttributeMask &AM) const;

  /// Return true if the builder has the specified attribute.
  LLVM_ABI bool contains(Attribute::AttrKind A) const;

  /// Return true if the builder has the specified target-dependent
  /// attribute.
  LLVM_ABI bool contains(StringRef A) const;

  /// Return true if the builder has IR-level attributes.
  bool hasAttributes() const { return !Attrs.empty(); }

  /// Return Attribute with the given Kind. The returned attribute will be
  /// invalid if the Kind is not present in the builder.
  LLVM_ABI Attribute getAttribute(Attribute::AttrKind Kind) const;

  /// Return Attribute with the given Kind. The returned attribute will be
  /// invalid if the Kind is not present in the builder.
  LLVM_ABI Attribute getAttribute(StringRef Kind) const;

  /// Retrieve the range if the attribute exists (std::nullopt is returned
  /// otherwise).
  LLVM_ABI std::optional<ConstantRange> getRange() const;

  /// Return raw (possibly packed/encoded) value of integer attribute or
  /// std::nullopt if not set.
  LLVM_ABI std::optional<uint64_t>
  getRawIntAttr(Attribute::AttrKind Kind) const;

  /// Retrieve the alignment attribute, if it exists.
  MaybeAlign getAlignment() const {
    return MaybeAlign(getRawIntAttr(Attribute::Alignment).value_or(0));
  }

  /// Retrieve the stack alignment attribute, if it exists.
  MaybeAlign getStackAlignment() const {
    return MaybeAlign(getRawIntAttr(Attribute::StackAlignment).value_or(0));
  }

  /// Retrieve the number of dereferenceable bytes, if the
  /// dereferenceable attribute exists (zero is returned otherwise).
  uint64_t getDereferenceableBytes() const {
    return getRawIntAttr(Attribute::Dereferenceable).value_or(0);
  }

  /// Retrieve the number of dereferenceable_or_null bytes, if the
  /// dereferenceable_or_null attribute exists (zero is returned otherwise).
  uint64_t getDereferenceableOrNullBytes() const {
    return getRawIntAttr(Attribute::DereferenceableOrNull).value_or(0);
  }

  /// Retrieve the bitmask for nofpclass, if the nofpclass attribute exists
  /// (fcNone is returned otherwise).
  FPClassTest getNoFPClass() const {
    std::optional<uint64_t> Raw = getRawIntAttr(Attribute::NoFPClass);
    return static_cast<FPClassTest>(Raw.value_or(0));
  }

  /// Retrieve type for the given type attribute.
  LLVM_ABI Type *getTypeAttr(Attribute::AttrKind Kind) const;

  /// Retrieve the byval type.
  Type *getByValType() const { return getTypeAttr(Attribute::ByVal); }

  /// Retrieve the sret type.
  Type *getStructRetType() const { return getTypeAttr(Attribute::StructRet); }

  /// Retrieve the byref type.
  Type *getByRefType() const { return getTypeAttr(Attribute::ByRef); }

  /// Retrieve the preallocated type.
  Type *getPreallocatedType() const {
    return getTypeAttr(Attribute::Preallocated);
  }

  /// Retrieve the inalloca type.
  Type *getInAllocaType() const { return getTypeAttr(Attribute::InAlloca); }

  /// Retrieve the allocsize args, or std::nullopt if the attribute does not
  /// exist.
  LLVM_ABI std::optional<std::pair<unsigned, std::optional<unsigned>>>
  getAllocSizeArgs() const;

  /// Add integer attribute with raw value (packed/encoded if necessary).
  LLVM_ABI AttrBuilder &addRawIntAttr(Attribute::AttrKind Kind, uint64_t Value);

  /// This turns an alignment into the form used internally in Attribute.
  /// This call has no effect if Align is not set.
  LLVM_ABI AttrBuilder &addAlignmentAttr(MaybeAlign Align);

  /// This turns an int alignment (which must be a power of 2) into the
  /// form used internally in Attribute.
  /// This call has no effect if Align is 0.
  /// Deprecated, use the version using a MaybeAlign.
  inline AttrBuilder &addAlignmentAttr(unsigned Align) {
    return addAlignmentAttr(MaybeAlign(Align));
  }

  /// This turns a stack alignment into the form used internally in Attribute.
  /// This call has no effect if Align is not set.
  LLVM_ABI AttrBuilder &addStackAlignmentAttr(MaybeAlign Align);

  /// This turns an int stack alignment (which must be a power of 2) into
  /// the form used internally in Attribute.
  /// This call has no effect if Align is 0.
  /// Deprecated, use the version using a MaybeAlign.
  inline AttrBuilder &addStackAlignmentAttr(unsigned Align) {
    return addStackAlignmentAttr(MaybeAlign(Align));
  }

  /// This turns the number of dereferenceable bytes into the form used
  /// internally in Attribute.
  LLVM_ABI AttrBuilder &addDereferenceableAttr(uint64_t Bytes);

  /// This turns the number of dereferenceable_or_null bytes into the
  /// form used internally in Attribute.
  LLVM_ABI AttrBuilder &addDereferenceableOrNullAttr(uint64_t Bytes);

  /// This turns one (or two) ints into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &
  addAllocSizeAttr(unsigned ElemSizeArg,
                   const std::optional<unsigned> &NumElemsArg);

  /// This turns two ints into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addVScaleRangeAttr(unsigned MinValue,
                                           std::optional<unsigned> MaxValue);

  /// Add a type attribute with the given type.
  LLVM_ABI AttrBuilder &addTypeAttr(Attribute::AttrKind Kind, Type *Ty);

  /// This turns a byval type into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addByValAttr(Type *Ty);

  /// This turns a sret type into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addStructRetAttr(Type *Ty);

  /// This turns a byref type into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addByRefAttr(Type *Ty);

  /// This turns a preallocated type into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addPreallocatedAttr(Type *Ty);

  /// This turns an inalloca type into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addInAllocaAttr(Type *Ty);

  /// Add an allocsize attribute, using the representation returned by
  /// Attribute.getIntValue().
  LLVM_ABI AttrBuilder &addAllocSizeAttrFromRawRepr(uint64_t RawAllocSizeRepr);

  /// Add a vscale_range attribute, using the representation returned by
  /// Attribute.getIntValue().
  LLVM_ABI AttrBuilder &
  addVScaleRangeAttrFromRawRepr(uint64_t RawVScaleRangeRepr);

  /// This turns the unwind table kind into the form used internally in
  /// Attribute.
  LLVM_ABI AttrBuilder &addUWTableAttr(UWTableKind Kind);

  // This turns the allocator kind into the form used internally in Attribute.
  LLVM_ABI AttrBuilder &addAllocKindAttr(AllocFnKind Kind);

  /// Add memory effect attribute.
  LLVM_ABI AttrBuilder &addMemoryAttr(MemoryEffects ME);

  /// Add captures attribute.
  LLVM_ABI AttrBuilder &addCapturesAttr(CaptureInfo CI);

  // Add nofpclass attribute
  LLVM_ABI AttrBuilder &addNoFPClassAttr(FPClassTest NoFPClassMask);

  /// Add a ConstantRange attribute with the given range.
  LLVM_ABI AttrBuilder &addConstantRangeAttr(Attribute::AttrKind Kind,
                                             const ConstantRange &CR);

  /// Add range attribute.
  LLVM_ABI AttrBuilder &addRangeAttr(const ConstantRange &CR);

  /// Add a ConstantRangeList attribute with the given ranges.
  LLVM_ABI AttrBuilder &addConstantRangeListAttr(Attribute::AttrKind Kind,
                                                 ArrayRef<ConstantRange> Val);

  /// Add initializes attribute.
  LLVM_ABI AttrBuilder &addInitializesAttr(const ConstantRangeList &CRL);

  /// Add 0 or more parameter attributes which are equivalent to metadata
  /// attached to \p I. e.g. !align -> align. This assumes the argument type is
  /// the same as the original instruction and the attribute is compatible.
  LLVM_ABI AttrBuilder &addFromEquivalentMetadata(const Instruction &I);

  ArrayRef<Attribute> attrs() const { return Attrs; }

  LLVM_ABI bool operator==(const AttrBuilder &B) const;
  bool operator!=(const AttrBuilder &B) const { return !(*this == B); }
};

namespace AttributeFuncs {

enum AttributeSafetyKind : uint8_t {
  ASK_SAFE_TO_DROP = 1,
  ASK_UNSAFE_TO_DROP = 2,
  ASK_ALL = ASK_SAFE_TO_DROP | ASK_UNSAFE_TO_DROP,
};

/// Returns true if this is a type legal for the 'nofpclass' attribute. This
/// follows the same type rules as FPMathOperator.
LLVM_ABI bool isNoFPClassCompatibleType(Type *Ty);

/// Which attributes cannot be applied to a type. The argument \p AS
/// is used as a hint for the attributes whose compatibility is being
/// checked against \p Ty. This does not mean the return will be a
/// subset of \p AS, just that attributes that have specific dynamic
/// type compatibilities (i.e `range`) will be checked against what is
/// contained in \p AS. The argument \p ASK indicates, if only
/// attributes that are known to be safely droppable are contained in
/// the mask; only attributes that might be unsafe to drop (e.g.,
/// ABI-related attributes) are in the mask; or both.
LLVM_ABI AttributeMask typeIncompatible(Type *Ty, AttributeSet AS,
                                        AttributeSafetyKind ASK = ASK_ALL);

/// Get param/return attributes which imply immediate undefined behavior if an
/// invalid value is passed. For example, this includes noundef (where undef
/// implies UB), but not nonnull (where null implies poison). It also does not
/// include attributes like nocapture, which constrain the function
/// implementation rather than the passed value.
LLVM_ABI AttributeMask getUBImplyingAttributes();

/// \returns Return true if the two functions have compatible target-independent
/// attributes for inlining purposes.
LLVM_ABI bool areInlineCompatible(const Function &Caller,
                                  const Function &Callee);

/// Checks  if there are any incompatible function attributes between
/// \p A and \p B.
///
/// \param [in] A - The first function to be compared with.
/// \param [in] B - The second function to be compared with.
/// \returns true if the functions have compatible attributes.
LLVM_ABI bool areOutlineCompatible(const Function &A, const Function &B);

/// Merge caller's and callee's attributes.
LLVM_ABI void mergeAttributesForInlining(Function &Caller,
                                         const Function &Callee);

/// Merges the functions attributes from \p ToMerge into function \p Base.
///
/// \param [in,out] Base - The function being merged into.
/// \param [in] ToMerge - The function to merge attributes from.
LLVM_ABI void mergeAttributesForOutlining(Function &Base,
                                          const Function &ToMerge);

/// Update min-legal-vector-width if it is in Attribute and less than Width.
LLVM_ABI void updateMinLegalVectorWidthAttr(Function &Fn, uint64_t Width);

} // end namespace AttributeFuncs

} // end namespace llvm

#endif // LLVM_IR_ATTRIBUTES_H
