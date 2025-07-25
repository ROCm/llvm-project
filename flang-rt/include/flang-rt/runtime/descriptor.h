//===-- include/flang-rt/runtime/descriptor.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_DESCRIPTOR_H_
#define FLANG_RT_RUNTIME_DESCRIPTOR_H_

// Defines data structures used during execution of a Fortran program
// to implement nontrivial dummy arguments, pointers, allocatables,
// function results, and the special behaviors of instances of derived types.
// This header file includes and extends the published language
// interoperability header that is required by the Fortran 2018 standard
// as a subset of definitions suitable for exposure to user C/C++ code.
// User C code is welcome to depend on that ISO_Fortran_binding.h file,
// but should never reference this internal header.

#include "memory.h"
#include "type-code.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/descriptor-consts.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

RT_OFFLOAD_VAR_GROUP_BEGIN
/// Value used for asyncId when no specific stream is specified.
static constexpr std::int64_t kNoAsyncId = -1;
/// Value used for asyncObject when no specific stream is specified.
static constexpr std::int64_t *kNoAsyncObject = nullptr;
RT_OFFLOAD_VAR_GROUP_END

namespace Fortran::runtime {

class Terminator;

RT_VAR_GROUP_BEGIN
static constexpr RT_CONST_VAR_ATTRS int maxRank{CFI_MAX_RANK};
RT_VAR_GROUP_END

// A C++ view of the sole interoperable standard descriptor (ISO::CFI_cdesc_t)
// and its type and per-dimension information.

class Dimension {
public:
  RT_API_ATTRS SubscriptValue LowerBound() const { return raw_.lower_bound; }
  RT_API_ATTRS SubscriptValue Extent() const { return raw_.extent; }
  RT_API_ATTRS SubscriptValue UpperBound() const {
    return LowerBound() + Extent() - 1;
  }
  RT_API_ATTRS SubscriptValue ByteStride() const { return raw_.sm; }

  RT_API_ATTRS Dimension &SetBounds(
      SubscriptValue lower, SubscriptValue upper) {
    if (upper >= lower) {
      raw_.lower_bound = lower;
      raw_.extent = upper - lower + 1;
    } else {
      raw_.lower_bound = 1;
      raw_.extent = 0;
    }
    return *this;
  }
  // Do not use this API to cause the LB of an empty dimension
  // to be anything other than 1.  Use SetBounds() instead if you can.
  RT_API_ATTRS Dimension &SetLowerBound(SubscriptValue lower) {
    raw_.lower_bound = lower;
    return *this;
  }
  RT_API_ATTRS Dimension &SetUpperBound(SubscriptValue upper) {
    auto lower{raw_.lower_bound};
    raw_.extent = upper >= lower ? upper - lower + 1 : 0;
    return *this;
  }
  RT_API_ATTRS Dimension &SetExtent(SubscriptValue extent) {
    raw_.extent = extent;
    return *this;
  }
  RT_API_ATTRS Dimension &SetByteStride(SubscriptValue bytes) {
    raw_.sm = bytes;
    return *this;
  }

private:
  ISO::CFI_dim_t raw_;
};

// The storage for this object follows the last used dim[] entry in a
// Descriptor (CFI_cdesc_t) generic descriptor.  Space matters here, since
// descriptors serve as POINTER and ALLOCATABLE components of derived type
// instances.  The presence of this structure is encoded in the
// CFI_cdesc_t.extra field, and the number of elements in the len_[]
// array is determined by derivedType_->LenParameters().
class DescriptorAddendum {
public:
  explicit RT_API_ATTRS DescriptorAddendum(
      const typeInfo::DerivedType *dt = nullptr)
      : derivedType_{dt}, len_{0} {}
  RT_API_ATTRS DescriptorAddendum &operator=(const DescriptorAddendum &);

  RT_API_ATTRS const typeInfo::DerivedType *derivedType() const {
    return derivedType_;
  }
  RT_API_ATTRS DescriptorAddendum &set_derivedType(
      const typeInfo::DerivedType *dt) {
    derivedType_ = dt;
    return *this;
  }

  RT_API_ATTRS std::size_t LenParameters() const;

  RT_API_ATTRS typeInfo::TypeParameterValue LenParameterValue(int which) const {
    return len_[which];
  }
  static constexpr RT_API_ATTRS std::size_t SizeInBytes(int lenParameters) {
    // TODO: Don't waste that last word if lenParameters == 0
    return sizeof(DescriptorAddendum) +
        std::max(lenParameters - 1, 0) * sizeof(typeInfo::TypeParameterValue);
  }
  RT_API_ATTRS std::size_t SizeInBytes() const;

  RT_API_ATTRS void SetLenParameterValue(
      int which, typeInfo::TypeParameterValue x) {
    len_[which] = x;
  }

  void Dump(FILE * = stdout) const;

private:
  const typeInfo::DerivedType *derivedType_;
  typeInfo::TypeParameterValue len_[1]; // must be the last component
  // The LEN type parameter values can also include captured values of
  // specification expressions that were used for bounds and for LEN type
  // parameters of components.  The values have been truncated to the LEN
  // type parameter's type, if shorter than 64 bits, then sign-extended.
};

// A C++ view of a standard descriptor object.
class Descriptor {
public:
  // Be advised: this class type is not suitable for use when allocating
  // a descriptor -- it is a dynamic view of the common descriptor format.
  // If used in a simple declaration of a local variable or dynamic allocation,
  // the size is going to be correct only by accident, since the true size of
  // a descriptor depends on the number of its dimensions and the presence and
  // size of an addendum, which depends on the type of the data.
  // Use the class template StaticDescriptor (below) to declare a descriptor
  // whose type and rank are fixed and known at compilation time.  Use the
  // Create() static member functions otherwise to dynamically allocate a
  // descriptor.

  RT_API_ATTRS Descriptor(const Descriptor &);
  RT_API_ATTRS Descriptor &operator=(const Descriptor &);

  // Returns the number of bytes occupied by an element of the given
  // category and kind including any alignment padding required
  // between adjacent elements.
  static RT_API_ATTRS std::size_t BytesFor(TypeCategory category, int kind);

  RT_API_ATTRS void Establish(TypeCode t, std::size_t elementBytes,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  RT_API_ATTRS void Establish(TypeCategory, int kind, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  RT_API_ATTRS void Establish(int characterKind, std::size_t characters,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  RT_API_ATTRS void Establish(const typeInfo::DerivedType &dt,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

  // To create a descriptor for a derived type the caller
  // must provide non-null dt argument.
  // The addendum argument is only used for testing purposes,
  // and it may force a descriptor with an addendum while
  // dt may be null.
  static RT_API_ATTRS OwningPtr<Descriptor> Create(TypeCode t,
      std::size_t elementBytes, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false, const typeInfo::DerivedType *dt = nullptr);
  static RT_API_ATTRS OwningPtr<Descriptor> Create(TypeCategory, int kind,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static RT_API_ATTRS OwningPtr<Descriptor> Create(int characterKind,
      SubscriptValue characters, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static RT_API_ATTRS OwningPtr<Descriptor> Create(
      const typeInfo::DerivedType &dt, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

  RT_API_ATTRS ISO::CFI_cdesc_t &raw() { return raw_; }
  RT_API_ATTRS const ISO::CFI_cdesc_t &raw() const { return raw_; }
  RT_API_ATTRS std::size_t ElementBytes() const { return raw_.elem_len; }
  RT_API_ATTRS int rank() const { return raw_.rank; }
  RT_API_ATTRS TypeCode type() const { return TypeCode{raw_.type}; }

  RT_API_ATTRS Descriptor &set_base_addr(void *p) {
    raw_.base_addr = p;
    return *this;
  }

  RT_API_ATTRS bool IsPointer() const {
    return raw_.attribute == CFI_attribute_pointer;
  }
  RT_API_ATTRS bool IsAllocatable() const {
    return raw_.attribute == CFI_attribute_allocatable;
  }
  RT_API_ATTRS bool IsAllocated() const { return raw_.base_addr != nullptr; }

  RT_API_ATTRS Dimension &GetDimension(int dim) {
    return *reinterpret_cast<Dimension *>(&raw_.dim[dim]);
  }
  RT_API_ATTRS const Dimension &GetDimension(int dim) const {
    return *reinterpret_cast<const Dimension *>(&raw_.dim[dim]);
  }

  RT_API_ATTRS std::size_t SubscriptByteOffset(
      int dim, SubscriptValue subscriptValue) const {
    const Dimension &dimension{GetDimension(dim)};
    return (subscriptValue - dimension.LowerBound()) * dimension.ByteStride();
  }

  RT_API_ATTRS std::size_t SubscriptsToByteOffset(
      const SubscriptValue subscript[]) const {
    std::size_t offset{0};
    for (int j{0}; j < raw_.rank; ++j) {
      offset += SubscriptByteOffset(j, subscript[j]);
    }
    return offset;
  }

  template <typename A = char>
  RT_API_ATTRS A *OffsetElement(std::size_t offset = 0) const {
    return reinterpret_cast<A *>(
        reinterpret_cast<char *>(raw_.base_addr) + offset);
  }

  template <typename A>
  RT_API_ATTRS A *Element(const SubscriptValue subscript[]) const {
    return OffsetElement<A>(SubscriptsToByteOffset(subscript));
  }

  template <typename A>
  RT_API_ATTRS A *ElementComponent(
      const SubscriptValue subscript[], std::size_t componentOffset) const {
    return OffsetElement<A>(
        SubscriptsToByteOffset(subscript) + componentOffset);
  }

  template <typename A>
  RT_API_ATTRS A *ZeroBasedIndexedElement(std::size_t n) const {
    if (raw_.rank == 0) {
      if (n == 0) {
        return OffsetElement<A>();
      }
    } else if (raw_.rank == 1) {
      const auto &dim{GetDimension(0)};
      if (n < static_cast<std::size_t>(dim.Extent())) {
        return OffsetElement<A>(n * dim.ByteStride());
      }
    } else {
      SubscriptValue at[maxRank];
      if (SubscriptsForZeroBasedElementNumber(at, n)) {
        return Element<A>(at);
      }
    }
    return nullptr;
  }

  RT_API_ATTRS int GetLowerBounds(SubscriptValue subscript[]) const {
    for (int j{0}; j < raw_.rank; ++j) {
      subscript[j] = GetDimension(j).LowerBound();
    }
    return raw_.rank;
  }

  RT_API_ATTRS int GetShape(SubscriptValue subscript[]) const {
    for (int j{0}; j < raw_.rank; ++j) {
      subscript[j] = GetDimension(j).Extent();
    }
    return raw_.rank;
  }

  // When the passed subscript vector contains the last (or first)
  // subscripts of the array, these wrap the subscripts around to
  // their first (or last) values and return false.
  RT_API_ATTRS bool IncrementSubscripts(
      SubscriptValue subscript[], const int *permutation = nullptr) const {
    for (int j{0}; j < raw_.rank; ++j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      if (subscript[k]++ < dim.UpperBound()) {
        return true;
      }
      subscript[k] = dim.LowerBound();
    }
    return false;
  }

  RT_API_ATTRS bool DecrementSubscripts(
      SubscriptValue[], const int *permutation = nullptr) const;

  // False when out of range.
  RT_API_ATTRS bool SubscriptsForZeroBasedElementNumber(
      SubscriptValue subscript[], std::size_t elementNumber,
      const int *permutation = nullptr) const {
    if (raw_.rank == 0) {
      return elementNumber == 0;
    }
    std::size_t dimCoefficient[maxRank];
    int k0{permutation ? permutation[0] : 0};
    dimCoefficient[0] = 1;
    auto coefficient{static_cast<std::size_t>(GetDimension(k0).Extent())};
    for (int j{1}; j < raw_.rank; ++j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      dimCoefficient[j] = coefficient;
      coefficient *= dim.Extent();
    }
    if (elementNumber >= coefficient) {
      return false; // out of range
    }
    for (int j{raw_.rank - 1}; j > 0; --j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      std::size_t quotient{elementNumber / dimCoefficient[j]};
      subscript[k] = quotient + dim.LowerBound();
      elementNumber -= quotient * dimCoefficient[j];
    }
    subscript[k0] = elementNumber + GetDimension(k0).LowerBound();
    return true;
  }

  RT_API_ATTRS std::size_t ZeroBasedElementNumber(
      const SubscriptValue *, const int *permutation = nullptr) const;

  RT_API_ATTRS DescriptorAddendum *Addendum() {
    if (HasAddendum()) {
      return reinterpret_cast<DescriptorAddendum *>(&GetDimension(rank()));
    } else {
      return nullptr;
    }
  }
  RT_API_ATTRS const DescriptorAddendum *Addendum() const {
    if (HasAddendum()) {
      return reinterpret_cast<const DescriptorAddendum *>(
          &GetDimension(rank()));
    } else {
      return nullptr;
    }
  }

  // Returns size in bytes of the descriptor (not the data)
  static constexpr RT_API_ATTRS std::size_t SizeInBytes(
      int rank, bool addendum = false, int lengthTypeParameters = 0) {
    std::size_t bytes{sizeof(Descriptor) - sizeof(Dimension)};
    bytes += rank * sizeof(Dimension);
    if (addendum || lengthTypeParameters > 0) {
      bytes += DescriptorAddendum::SizeInBytes(lengthTypeParameters);
    }
    return bytes;
  }

  RT_API_ATTRS std::size_t SizeInBytes() const;

  RT_API_ATTRS std::size_t Elements() const;
  RT_API_ATTRS std::size_t InlineElements() const {
    int n{rank()};
    if (n == 0) {
      return 1;
    } else {
      auto elements{static_cast<std::size_t>(GetDimension(0).Extent())};
      for (int j{1}; j < n; ++j) {
        elements *= GetDimension(j).Extent();
      }
      return elements;
    }
  }

  // Allocate() assumes Elements() and ElementBytes() work;
  // define the extents of the dimensions and the element length
  // before calling.  It (re)computes the byte strides after
  // allocation.  Does not allocate automatic components or
  // perform default component initialization.
  RT_API_ATTRS int Allocate(std::int64_t *asyncObject);
  RT_API_ATTRS void SetByteStrides();

  // Deallocates storage; does not call FINAL subroutines or
  // deallocate allocatable/automatic components.
  RT_API_ATTRS int Deallocate() {
    ISO::CFI_cdesc_t &descriptor{raw()};
    void *pointer{descriptor.base_addr};
    if (!pointer) {
      return CFI_ERROR_BASE_ADDR_NULL;
    } else {
      int allocIndex{MapAllocIdx()};
      if (allocIndex == kDefaultAllocator) {
        std::free(pointer);
      } else {
        allocatorRegistry.GetDeallocator(MapAllocIdx())(pointer);
      }
      descriptor.base_addr = nullptr;
      return CFI_SUCCESS;
    }
  }

  // Deallocates storage, including allocatable and automatic
  // components.  Optionally invokes FINAL subroutines.
  RT_API_ATTRS int Destroy(bool finalize = false, bool destroyPointers = false,
      Terminator * = nullptr);

  RT_API_ATTRS bool IsContiguous(int leadingDimensions = maxRank) const {
    auto bytes{static_cast<SubscriptValue>(ElementBytes())};
    if (leadingDimensions > raw_.rank) {
      leadingDimensions = raw_.rank;
    }
    bool stridesAreContiguous{true};
    for (int j{0}; j < leadingDimensions; ++j) {
      const Dimension &dim{GetDimension(j)};
      stridesAreContiguous &= bytes == dim.ByteStride() || dim.Extent() == 1;
      bytes *= dim.Extent();
    }
    // One and zero element arrays are contiguous even if the descriptor
    // byte strides are not perfect multiples.
    // Arrays with more than 2 elements may also be contiguous even if a
    // byte stride in one dimension is not a perfect multiple, as long as
    // this is the last dimension, or if the dimension has one extent and
    // the following dimension have either one extents or contiguous byte
    // strides.
    return stridesAreContiguous || bytes == 0;
  }

  // The result, if any, is a fixed stride value that can be used to
  // address all elements.  It generalizes contiguity by also allowing
  // the case of an array with extent 1 on all but one dimension.
  RT_API_ATTRS common::optional<SubscriptValue> FixedStride() const {
    auto rank{static_cast<std::size_t>(raw_.rank)};
    common::optional<SubscriptValue> stride;
    for (std::size_t j{0}; j < rank; ++j) {
      const Dimension &dim{GetDimension(j)};
      auto extent{dim.Extent()};
      if (extent == 0) {
        break; // empty array
      } else if (extent == 1) { // ok
      } else if (stride) {
        // Extent > 1 on multiple dimensions
        if (IsContiguous()) {
          return ElementBytes();
        } else {
          return common::nullopt;
        }
      } else {
        stride = dim.ByteStride();
      }
    }
    return stride.value_or(0); // 0 for scalars and empty arrays
  }

  // Establishes a pointer to a section or element.
  RT_API_ATTRS bool EstablishPointerSection(const Descriptor &source,
      const SubscriptValue *lower = nullptr,
      const SubscriptValue *upper = nullptr,
      const SubscriptValue *stride = nullptr);

  RT_API_ATTRS void ApplyMold(
      const Descriptor &, int rank, bool isMonomorphic = false);

  RT_API_ATTRS void Check() const;

  void Dump(FILE * = stdout) const;

  RT_API_ATTRS inline bool HasAddendum() const {
    return raw_.extra & _CFI_ADDENDUM_FLAG;
  }
  RT_API_ATTRS inline void SetHasAddendum() {
    raw_.extra |= _CFI_ADDENDUM_FLAG;
  }
  RT_API_ATTRS inline int GetAllocIdx() const {
    return (raw_.extra & _CFI_ALLOCATOR_IDX_MASK) >> _CFI_ALLOCATOR_IDX_SHIFT;
  }
  RT_API_ATTRS int MapAllocIdx() const {
#ifdef RT_DEVICE_COMPILATION
    // Force default allocator in device code.
    return kDefaultAllocator;
#else
    return GetAllocIdx();
#endif
  }
  RT_API_ATTRS inline void SetAllocIdx(int pos) {
    raw_.extra &= ~_CFI_ALLOCATOR_IDX_MASK; // Clear the allocator index bits.
    raw_.extra |= pos << _CFI_ALLOCATOR_IDX_SHIFT;
  }

private:
  ISO::CFI_cdesc_t raw_;
};
static_assert(sizeof(Descriptor) == sizeof(ISO::CFI_cdesc_t));

// Lightweight iterator-like API to simplify specialising Descriptor indexing
// in cases where it can improve application performance. On account of the
// purpose of this API being performance optimisation, it is up to the user to
// do all the necessary checks to make sure the specialised variants can be used
// safely and that Advance() is not called more times than the number of
// elements in the Descriptor allows for.
// Default RANK=-1 supports aray descriptors of any rank up to maxRank.
template <int RANK = -1> class DescriptorIterator {
private:
  const Descriptor &descriptor;
  SubscriptValue subscripts[maxRank];
  std::size_t elementOffset{0};

public:
  RT_API_ATTRS DescriptorIterator(const Descriptor &descriptor)
      : descriptor(descriptor) {
    // We do not need the subscripts to iterate over a rank-1 array
    if constexpr (RANK != 1) {
      descriptor.GetLowerBounds(subscripts);
    }
  };

  template <typename A> RT_API_ATTRS A *Get() {
    std::size_t offset{0};
    // The rank-1 case doesn't require looping at all
    if constexpr (RANK == 1) {
      offset = elementOffset;
      // The compiler might be able to optimise this better if we know the rank
      // at compile time
    } else if constexpr (RANK != -1) {
      for (int j{0}; j < RANK; ++j) {
        offset += descriptor.SubscriptByteOffset(j, subscripts[j]);
      }
      // General fallback
    } else {
      offset = descriptor.SubscriptsToByteOffset(subscripts);
    }

    return descriptor.OffsetElement<A>(offset);
  }

  RT_API_ATTRS void Advance() {
    if constexpr (RANK == 1) {
      elementOffset += descriptor.GetDimension(0).ByteStride();
    } else if constexpr (RANK != -1) {
      for (int j{0}; j < RANK; ++j) {
        const Dimension &dim{descriptor.GetDimension(j)};
        if (subscripts[j]++ < dim.UpperBound()) {
          break;
        }
        subscripts[j] = dim.LowerBound();
      }
    } else {
      descriptor.IncrementSubscripts(subscripts);
    }
  }
};

// Properly configured instances of StaticDescriptor will occupy the
// exact amount of storage required for the descriptor, its dimensional
// information, and possible addendum.  To build such a static descriptor,
// declare an instance of StaticDescriptor<>, extract a reference to its
// descriptor via the descriptor() accessor, and then built a Descriptor
// therein via descriptor.Establish(), e.g.:
//   StaticDescriptor<R,A,LP> statDesc;
//   Descriptor &descriptor{statDesc.descriptor()};
//   descriptor.Establish( ... );
template <int MAX_RANK = maxRank, bool ADDENDUM = false, int MAX_LEN_PARMS = 0>
class alignas(Descriptor) StaticDescriptor {
public:
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr int maxRank{MAX_RANK};
  static constexpr int maxLengthTypeParameters{MAX_LEN_PARMS};
  static constexpr bool hasAddendum{ADDENDUM || MAX_LEN_PARMS > 0};
  static constexpr std::size_t byteSize{
      Descriptor::SizeInBytes(maxRank, hasAddendum, maxLengthTypeParameters)};
  static_assert(byteSize <=
      MaxDescriptorSizeInBytes(maxRank, hasAddendum, maxLengthTypeParameters));
  RT_OFFLOAD_VAR_GROUP_END

  RT_API_ATTRS Descriptor &descriptor() {
    return *reinterpret_cast<Descriptor *>(storage_);
  }
  RT_API_ATTRS const Descriptor &descriptor() const {
    return *reinterpret_cast<const Descriptor *>(storage_);
  }

  RT_API_ATTRS void Check() {
    assert(descriptor().rank() <= maxRank);
    assert(descriptor().SizeInBytes() <= byteSize);
    if (DescriptorAddendum * addendum{descriptor().Addendum()}) {
      (void)addendum;
      assert(hasAddendum);
      assert(addendum->LenParameters() <= maxLengthTypeParameters);
    } else {
      assert(!hasAddendum);
      assert(maxLengthTypeParameters == 0);
    }
    descriptor().Check();
  }

private:
  char storage_[byteSize]{};
};

// Deduction guide to avoid warnings from older versions of clang.
StaticDescriptor() -> StaticDescriptor<maxRank, false, 0>;

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_DESCRIPTOR_H_
