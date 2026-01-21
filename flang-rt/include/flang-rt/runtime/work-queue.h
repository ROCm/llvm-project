//===-- include/flang-rt/runtime/work-queue.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Internal runtime utilities for work queues that replace the use of recursion
// for better GPU device support.
//
// A work queue comprises a list of tickets.  Each ticket class has a Begin()
// member function, which is called once, and a Continue() member function
// that can be called zero or more times.  A ticket's execution terminates
// when either of these member functions returns a status other than
// StatContinue.  When that status is not StatOk, then the whole queue
// is shut down.
//
// By returning StatContinue from its Continue() member function,
// a ticket suspends its execution so that any nested tickets that it
// may have created can be run to completion.  It is the reponsibility
// of each ticket class to maintain resumption information in its state
// and manage its own progress.  Most ticket classes inherit from
// class ComponentsOverElements, which implements an outer loop over all
// components of a derived type, and an inner loop over all elements
// of a descriptor, possibly with multiple phases of execution per element.
//
// Tickets are created by WorkQueue::Begin...() member functions.
// There is one of these for each "top level" recursive function in the
// Fortran runtime support library that has been restructured into this
// ticket framework.
//
// When the work queue is running tickets, it always selects the last ticket
// on the list for execution -- "work stack" might have been a more accurate
// name for this framework.  This ticket may, while doing its job, create
// new tickets, and since those are pushed after the active one, the first
// such nested ticket will be the next one executed to completion -- i.e.,
// the order of nested WorkQueue::Begin...() calls is respected.
// Note that a ticket's Continue() member function won't be called again
// until all nested tickets have run to completion and it is once again
// the last ticket on the queue.
//
// Example for an assignment to a derived type:
// 1. Assign() is called, and its work queue is created.  It calls
//    WorkQueue::BeginAssign() and then WorkQueue::Run().
// 2. Run calls AssignTicket::Begin(), which pushes a tickets via
//    BeginFinalize() and returns StatContinue.
// 3. FinalizeTicket::Begin() and FinalizeTicket::Continue() are called
//    until one of them returns StatOk, which ends the finalization ticket.
// 4. AssignTicket::Continue() is then called; it creates a DerivedAssignTicket
//    and then returns StatOk, which ends the ticket.
// 5. At this point, only one ticket remains.  DerivedAssignTicket::Begin()
//    and ::Continue() are called until they are done (not StatContinue).
//    Along the way, it may create nested AssignTickets for components,
//    and suspend itself so that they may each run to completion.

#ifndef FLANG_RT_RUNTIME_WORK_QUEUE_H_
#define FLANG_RT_RUNTIME_WORK_QUEUE_H_

#include "flang-rt/runtime/connection.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Common/api-attrs.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/freestanding-tools.h"
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace Fortran::runtime::io {
class IoStatementState;
struct NonTbpDefinedIoTable;
} // namespace Fortran::runtime::io

namespace Fortran::runtime {
class Terminator;
class WorkQueue;

RT_OFFLOAD_API_GROUP_BEGIN

// Ticket worker base classes

template <typename TICKET> class ImmediateTicketRunner {
public:
  RT_API_ATTRS explicit ImmediateTicketRunner(TICKET &ticket)
      : ticket_{ticket} {}
  RT_API_ATTRS int Run(WorkQueue &workQueue) {
    int status{ticket_.Begin(workQueue)};
    while (status == StatContinue) {
      status = ticket_.Continue(workQueue);
    }
    return status;
  }

private:
  TICKET &ticket_;
};

// Base class for ticket workers that operate elementwise over descriptors
class Elementwise {
public:
  RT_API_ATTRS Elementwise(
      const Descriptor &instance, const Descriptor *from = nullptr)
      : instance_{instance}, from_{from} {
    instance_.GetLowerBounds(subscripts_);
    if (from_) {
      from_->GetLowerBounds(fromSubscripts_);
    }
  }
  RT_API_ATTRS bool IsComplete() const { return elementAt_ >= elements_; }
  RT_API_ATTRS void Advance() {
    ++elementAt_;
    instance_.IncrementSubscripts(subscripts_);
    if (from_) {
      from_->IncrementSubscripts(fromSubscripts_);
    }
  }
  RT_API_ATTRS void SkipToEnd() { elementAt_ = elements_; }
  RT_API_ATTRS void Reset() {
    elementAt_ = 0;
    instance_.GetLowerBounds(subscripts_);
    if (from_) {
      from_->GetLowerBounds(fromSubscripts_);
    }
  }

protected:
  const Descriptor &instance_, *from_{nullptr};
  std::size_t elements_{instance_.InlineElements()};
  std::size_t elementAt_{0};
  SubscriptValue subscripts_[maxRank];
  SubscriptValue fromSubscripts_[maxRank];
};

// Base class for ticket workers that operate over derived type components.
class Componentwise {
public:
  RT_API_ATTRS Componentwise(const typeInfo::DerivedType &derived)
      : derived_{derived}, components_{derived_.component().InlineElements()} {
    GetFirstComponent();
  }

  RT_API_ATTRS bool IsComplete() const { return componentAt_ >= components_; }
  RT_API_ATTRS void Advance() {
    ++componentAt_;
    if (IsComplete()) {
      component_ = nullptr;
    } else {
      ++component_;
    }
  }
  RT_API_ATTRS void SkipToEnd() {
    component_ = nullptr;
    componentAt_ = components_;
  }
  RT_API_ATTRS void Reset() {
    component_ = nullptr;
    componentAt_ = 0;
    GetFirstComponent();
  }

protected:
  const typeInfo::DerivedType &derived_;
  std::size_t components_{0}, componentAt_{0};
  const typeInfo::Component *component_{nullptr};
  StaticDescriptor<maxRank, true, 0> componentDescriptor_;

private:
  RT_API_ATTRS void GetFirstComponent() {
    if (components_ > 0) {
      component_ = derived_.component().OffsetElement<typeInfo::Component>();
    }
  }
};

// Base class for ticket workers that operate over derived type components
// in an outer loop, and elements in an inner loop.
class ComponentsOverElements : public Componentwise, public Elementwise {
public:
  RT_API_ATTRS ComponentsOverElements(const Descriptor &instance,
      const typeInfo::DerivedType &derived, const Descriptor *from = nullptr)
      : Componentwise{derived}, Elementwise{instance, from} {
    if (Elementwise::IsComplete()) {
      Componentwise::SkipToEnd();
    }
  }
  RT_API_ATTRS bool IsComplete() const { return Componentwise::IsComplete(); }
  RT_API_ATTRS void Advance() {
    SkipToNextElement();
    if (Elementwise::IsComplete()) {
      Elementwise::Reset();
      Componentwise::Advance();
    }
  }
  RT_API_ATTRS void SkipToNextElement() {
    phase_ = 0;
    Elementwise::Advance();
  }
  RT_API_ATTRS void SkipToNextComponent() {
    phase_ = 0;
    Elementwise::Reset();
    Componentwise::Advance();
  }
  RT_API_ATTRS void Reset() {
    phase_ = 0;
    Elementwise::Reset();
    Componentwise::Reset();
  }

protected:
  int phase_{0};
};

// Base class for ticket workers that operate over elements in an outer loop,
// type components in an inner loop.
class ElementsOverComponents : public Elementwise, public Componentwise {
public:
  RT_API_ATTRS ElementsOverComponents(const Descriptor &instance,
      const typeInfo::DerivedType &derived, const Descriptor *from = nullptr)
      : Elementwise{instance, from}, Componentwise{derived} {
    if (Componentwise::IsComplete()) {
      Elementwise::SkipToEnd();
    }
  }
  RT_API_ATTRS bool IsComplete() const { return Elementwise::IsComplete(); }
  RT_API_ATTRS void Advance() {
    SkipToNextComponent();
    if (Componentwise::IsComplete()) {
      Componentwise::Reset();
      Elementwise::Advance();
    }
  }
  RT_API_ATTRS void SkipToNextComponent() {
    phase_ = 0;
    Componentwise::Advance();
  }
  RT_API_ATTRS void SkipToNextElement() {
    phase_ = 0;
    Componentwise::Reset();
    Elementwise::Advance();
  }

protected:
  int phase_{0};
};

// Unified base class that can operate in either componentwise or elementwise
// mode, selected at runtime. This allows a single DerivedAssignTicket class
// instead of two template instantiations.
class ComponentsAndElements : public Componentwise, public Elementwise {
public:
  RT_API_ATTRS ComponentsAndElements(const Descriptor &instance,
      const typeInfo::DerivedType &derived, const Descriptor *from,
      bool isComponentwise)
      : Componentwise{derived}, Elementwise{instance, from},
        isComponentwise_{isComponentwise} {
    if (isComponentwise_) {
      if (Elementwise::IsComplete()) {
        Componentwise::SkipToEnd();
      }
    } else {
      if (Componentwise::IsComplete()) {
        Elementwise::SkipToEnd();
      }
    }
  }
  RT_API_ATTRS bool IsComplete() const {
    return isComponentwise_ ? Componentwise::IsComplete()
                            : Elementwise::IsComplete();
  }
  RT_API_ATTRS void Advance() {
    if (isComponentwise_) {
      // ComponentsOverElements: outer loop over components, inner over elements
      SkipToNextElement();
      if (Elementwise::IsComplete()) {
        Elementwise::Reset();
        Componentwise::Advance();
      }
    } else {
      // ElementsOverComponents: outer loop over elements, inner over components
      SkipToNextComponent();
      if (Componentwise::IsComplete()) {
        Componentwise::Reset();
        Elementwise::Advance();
      }
    }
  }
  RT_API_ATTRS void SkipToNextElement() {
    phase_ = 0;
    if (isComponentwise_) {
      Elementwise::Advance();
    } else {
      Componentwise::Reset();
      Elementwise::Advance();
    }
  }
  RT_API_ATTRS void SkipToNextComponent() {
    phase_ = 0;
    if (isComponentwise_) {
      Elementwise::Reset();
      Componentwise::Advance();
    } else {
      Componentwise::Advance();
    }
  }
  RT_API_ATTRS void Reset() {
    phase_ = 0;
    Elementwise::Reset();
    Componentwise::Reset();
  }
  RT_API_ATTRS bool isComponentwise() const { return isComponentwise_; }

protected:
  int phase_{0};

private:
  bool isComponentwise_;
};

// Ticket worker classes

// Implements derived type instance initialization.
class InitializeTicket : public ImmediateTicketRunner<InitializeTicket>,
                         private ElementsOverComponents {
public:
  RT_API_ATTRS InitializeTicket(const Descriptor &instance,
      const typeInfo::DerivedType &derived, MemcpyFct memcpyFct)
      : ImmediateTicketRunner<InitializeTicket>{*this},
        ElementsOverComponents{instance, derived}, memcpyFct_{memcpyFct} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  MemcpyFct memcpyFct_;
};

// Initializes one derived type instance from the value of another
class InitializeCloneTicket
    : public ImmediateTicketRunner<InitializeCloneTicket>,
      private ComponentsOverElements {
public:
  RT_API_ATTRS InitializeCloneTicket(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg)
      : ImmediateTicketRunner<InitializeCloneTicket>{*this},
        ComponentsOverElements{original, derived}, clone_{clone},
        hasStat_{hasStat}, errMsg_{errMsg} {}
  RT_API_ATTRS int Begin(WorkQueue &) { return StatContinue; }
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const Descriptor &clone_;
  bool hasStat_{false};
  const Descriptor *errMsg_{nullptr};
  StaticDescriptor<maxRank, true, 0> cloneComponentDescriptor_;
};

// Implements derived type instance finalization
class FinalizeTicket : public ImmediateTicketRunner<FinalizeTicket>,
                       private ComponentsOverElements {
public:
  RT_API_ATTRS FinalizeTicket(
      const Descriptor &instance, const typeInfo::DerivedType &derived)
      : ImmediateTicketRunner<FinalizeTicket>{*this},
        ComponentsOverElements{instance, derived} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  const typeInfo::DerivedType *finalizableParentType_{nullptr};
};

// Implements derived type instance destruction
class DestroyTicket : public ImmediateTicketRunner<DestroyTicket>,
                      private ComponentsOverElements {
public:
  RT_API_ATTRS DestroyTicket(const Descriptor &instance,
      const typeInfo::DerivedType &derived, bool finalize)
      : ImmediateTicketRunner<DestroyTicket>{*this},
        ComponentsOverElements{instance, derived}, finalize_{finalize},
        fixedStride_{instance.FixedStride()} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  bool finalize_{false};
  common::optional<SubscriptValue> fixedStride_;
};

// Implements general intrinsic assignment
class AssignTicket : public ImmediateTicketRunner<AssignTicket> {
public:
  RT_API_ATTRS AssignTicket(Descriptor &to, const Descriptor &from, int flags,
      MemmoveFct memmoveFct, const typeInfo::DerivedType *declaredType)
      : ImmediateTicketRunner<AssignTicket>{*this}, to_{to}, from_{&from},
        flags_{flags}, memmoveFct_{memmoveFct}, declaredType_{declaredType} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  RT_API_ATTRS Descriptor &GetTempDescriptor();
  RT_API_ATTRS bool IsSimpleMemmove() const {
    return !toDerived_ && to_.rank() == from_->rank() && to_.IsContiguous() &&
        from_->IsContiguous() && to_.ElementBytes() == from_->ElementBytes();
  }

  Descriptor &to_;
  const Descriptor *from_{nullptr};
  int flags_{0}; // enum AssignFlags
  MemmoveFct memmoveFct_{nullptr};
  StaticDescriptor<maxRank, true, 0> tempDescriptor_;
  const typeInfo::DerivedType *declaredType_{nullptr};
  const typeInfo::DerivedType *toDerived_{nullptr};
  Descriptor *toDeallocate_{nullptr};
  bool persist_{false};
  bool done_{false};
};

// Implements derived type intrinsic assignment.
// Uses runtime flag instead of template parameter to reduce code size.
class DerivedAssignTicket : public ImmediateTicketRunner<DerivedAssignTicket>,
                            private ComponentsAndElements {
public:
  RT_API_ATTRS DerivedAssignTicket(const Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter, bool isComponentwise)
      : ImmediateTicketRunner<DerivedAssignTicket>{*this},
        ComponentsAndElements{to, derived, &from, isComponentwise},
        flags_{flags}, memmoveFct_{memmoveFct},
        deallocateAfter_{deallocateAfter} {}
  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  bool toIsContiguous_{this->instance_.IsContiguous()};
  bool fromIsContiguous_{this->from_->IsContiguous()};
  int flags_{0};
  MemmoveFct memmoveFct_{nullptr};
  Descriptor *deallocateAfter_{nullptr};
  StaticDescriptor<maxRank, true, 0> fromComponentDescriptor_;
};

// IO tickets are only used on host - descriptor-io.cpp is not compiled for GPU
#if !defined(RT_DEVICE_COMPILATION)
namespace io::descr {

template <io::Direction DIR>
class DescriptorIoTicket
    : public ImmediateTicketRunner<DescriptorIoTicket<DIR>>,
      private Elementwise {
public:
  RT_API_ATTRS DescriptorIoTicket(io::IoStatementState &io,
      const Descriptor &descriptor, const io::NonTbpDefinedIoTable *table,
      bool &anyIoTookPlace)
      : ImmediateTicketRunner<DescriptorIoTicket>(*this),
        Elementwise{descriptor}, io_{io}, table_{table},
        anyIoTookPlace_{anyIoTookPlace} {}

  RT_API_ATTRS int Begin(WorkQueue &);
  RT_API_ATTRS int Continue(WorkQueue &);
  RT_API_ATTRS bool &anyIoTookPlace() { return anyIoTookPlace_; }

private:
  io::IoStatementState &io_;
  const io::NonTbpDefinedIoTable *table_{nullptr};
  bool &anyIoTookPlace_;
  common::optional<typeInfo::SpecialBinding> nonTbpSpecial_;
  const typeInfo::DerivedType *derived_{nullptr};
  const typeInfo::SpecialBinding *special_{nullptr};
  StaticDescriptor<maxRank, true, 0> elementDescriptor_;
};

template <io::Direction DIR>
class DerivedIoTicket : public ImmediateTicketRunner<DerivedIoTicket<DIR>>,
                        private ElementsOverComponents {
public:
  RT_API_ATTRS DerivedIoTicket(io::IoStatementState &io,
      const Descriptor &descriptor, const typeInfo::DerivedType &derived,
      const io::NonTbpDefinedIoTable *table, bool &anyIoTookPlace)
      : ImmediateTicketRunner<DerivedIoTicket>(*this),
        ElementsOverComponents{descriptor, derived}, io_{io}, table_{table},
        anyIoTookPlace_{anyIoTookPlace} {}
  RT_API_ATTRS int Begin(WorkQueue &) { return StatContinue; }
  RT_API_ATTRS int Continue(WorkQueue &);

private:
  io::IoStatementState &io_;
  const io::NonTbpDefinedIoTable *table_{nullptr};
  bool &anyIoTookPlace_;
};

} // namespace io::descr
#endif // !defined(RT_DEVICE_COMPILATION)

struct NullTicket {
  RT_API_ATTRS int Begin(WorkQueue &) const { return StatOk; }
  RT_API_ATTRS int Continue(WorkQueue &) const { return StatOk; }
};

// Ticket type enumeration for tagged union
enum class TicketType : std::uint8_t {
  Null = 0,
  Initialize,
  InitializeClone,
  Finalize,
  Destroy,
  Assign,
  DerivedAssign, // Consolidated from DerivedAssignFalse/DerivedAssignTrue
#if !defined(RT_DEVICE_COMPILATION)
  DescriptorIoOutput,
  DescriptorIoInput,
  DerivedIoOutput,
  DerivedIoInput
#endif
};

// Helper template to calculate maximum size at compile time
template <std::size_t A, std::size_t B> struct MaxSize {
  static constexpr std::size_t value = (A > B) ? A : B;
};

template <std::size_t A, std::size_t B, std::size_t C, std::size_t D,
    std::size_t E, std::size_t F, std::size_t G>
struct MaxSize7 {
  static constexpr std::size_t value = MaxSize<A,
      MaxSize<B,
          MaxSize<C,
              MaxSize<D,
                  MaxSize<E, MaxSize<F, G>::value>::value>::value>::value>::
              value>::value;
};

#if !defined(RT_DEVICE_COMPILATION)
template <std::size_t A, std::size_t B, std::size_t C, std::size_t D,
    std::size_t E, std::size_t F, std::size_t G, std::size_t H, std::size_t I,
    std::size_t J, std::size_t K>
struct MaxSize11 {
  static constexpr std::size_t value = MaxSize<A,
      MaxSize<B,
          MaxSize<C,
              MaxSize<D,
                  MaxSize<E,
                      MaxSize<F,
                          MaxSize<G,
                              MaxSize<H,
                                  MaxSize<I, MaxSize<J, K>::value>::value>::
                                      value>::value>::value>::value>::value>::
                          value>::value>::value;
};
#endif

// Forward declarations for ticket storage
struct TicketStorage {
  // Calculate maximum size needed for any ticket type
  // We need to ensure proper alignment - use the maximum alignment required
#if !defined(RT_DEVICE_COMPILATION)
  static constexpr std::size_t maxSize_ = MaxSize11<
      sizeof(NullTicket), sizeof(InitializeTicket),
      sizeof(InitializeCloneTicket), sizeof(FinalizeTicket),
      sizeof(DestroyTicket), sizeof(AssignTicket),
      sizeof(DerivedAssignTicket),
      sizeof(io::descr::DescriptorIoTicket<io::Direction::Output>),
      sizeof(io::descr::DescriptorIoTicket<io::Direction::Input>),
      sizeof(io::descr::DerivedIoTicket<io::Direction::Output>),
      sizeof(io::descr::DerivedIoTicket<io::Direction::Input>)>::value;

  // Use maximum alignment - typically 8 or 16 bytes should be sufficient
  // but we'll use alignas with the largest alignment requirement
  alignas(alignof(InitializeTicket)) alignas(alignof(InitializeCloneTicket))
      alignas(alignof(FinalizeTicket)) alignas(alignof(DestroyTicket))
      alignas(alignof(AssignTicket)) alignas(alignof(DerivedAssignTicket))
      alignas(alignof(io::descr::DescriptorIoTicket<io::Direction::Output>))
      alignas(alignof(io::descr::DescriptorIoTicket<io::Direction::Input>))
      alignas(alignof(io::descr::DerivedIoTicket<io::Direction::Output>))
      alignas(alignof(io::descr::DerivedIoTicket<io::Direction::Input>))
      char storage[maxSize_];
#else
  // Device builds exclude IO tickets for reduced code size
  static constexpr std::size_t maxSize_ = MaxSize7<
      sizeof(NullTicket), sizeof(InitializeTicket),
      sizeof(InitializeCloneTicket), sizeof(FinalizeTicket),
      sizeof(DestroyTicket), sizeof(AssignTicket),
      sizeof(DerivedAssignTicket)>::value;

  alignas(alignof(InitializeTicket)) alignas(alignof(InitializeCloneTicket))
      alignas(alignof(FinalizeTicket)) alignas(alignof(DestroyTicket))
      alignas(alignof(AssignTicket)) alignas(alignof(DerivedAssignTicket))
      char storage[maxSize_];
#endif

  RT_API_ATTRS void *GetPtr() { return storage; }
  RT_API_ATTRS const void *GetPtr() const { return storage; }
};

struct Ticket {
  RT_API_ATTRS int Continue(WorkQueue &);
  RT_API_ATTRS ~Ticket();
  RT_API_ATTRS Ticket();
  RT_API_ATTRS Ticket(const Ticket &) = delete;
  RT_API_ATTRS Ticket &operator=(const Ticket &) = delete;
  RT_API_ATTRS Ticket(Ticket &&) = delete;
  RT_API_ATTRS Ticket &operator=(Ticket &&) = delete;

  // Template method to construct a ticket in place
  template <typename T, typename... Args>
  RT_API_ATTRS void emplace(Args &&...args) {
    // Destroy existing ticket if any
    destroy();
    // Set type
    type_ = getTicketType<T>();
    // Construct new ticket using placement new
    new (storage_.GetPtr()) T(std::forward<Args>(args)...);
  }

  // Get ticket type index (for debugging)
  RT_API_ATTRS std::size_t index() const {
    return static_cast<std::size_t>(type_);
  }

  // Get ticket type
  RT_API_ATTRS TicketType type() const { return type_; }

  bool begun{false};  // Public for WorkQueue access

private:
  RT_API_ATTRS void destroy();
  RT_API_ATTRS int dispatchBegin(WorkQueue &workQueue);
  RT_API_ATTRS int dispatchContinue(WorkQueue &workQueue);

  template <typename T> static constexpr TicketType getTicketType() {
    if constexpr (std::is_same_v<T, NullTicket>) {
      return TicketType::Null;
    } else if constexpr (std::is_same_v<T, InitializeTicket>) {
      return TicketType::Initialize;
    } else if constexpr (std::is_same_v<T, InitializeCloneTicket>) {
      return TicketType::InitializeClone;
    } else if constexpr (std::is_same_v<T, FinalizeTicket>) {
      return TicketType::Finalize;
    } else if constexpr (std::is_same_v<T, DestroyTicket>) {
      return TicketType::Destroy;
    } else if constexpr (std::is_same_v<T, AssignTicket>) {
      return TicketType::Assign;
    } else if constexpr (std::is_same_v<T, DerivedAssignTicket>) {
      return TicketType::DerivedAssign;
    }
#if !defined(RT_DEVICE_COMPILATION)
    else if constexpr (std::is_same_v<T,
                          io::descr::DescriptorIoTicket<io::Direction::Output>>) {
      return TicketType::DescriptorIoOutput;
    } else if constexpr (std::is_same_v<T,
                          io::descr::DescriptorIoTicket<io::Direction::Input>>) {
      return TicketType::DescriptorIoInput;
    } else if constexpr (std::is_same_v<T,
                          io::descr::DerivedIoTicket<io::Direction::Output>>) {
      return TicketType::DerivedIoOutput;
    } else if constexpr (std::is_same_v<T,
                          io::descr::DerivedIoTicket<io::Direction::Input>>) {
      return TicketType::DerivedIoInput;
    }
#endif
  }

  TicketType type_{TicketType::Null};
  TicketStorage storage_;
};

class WorkQueue {
public:
  RT_API_ATTRS explicit WorkQueue(Terminator &terminator)
      : terminator_{terminator} {
    for (int j{1}; j < numStatic_; ++j) {
      static_[j].previous = &static_[j - 1];
      static_[j - 1].next = &static_[j];
    }
  }
  RT_API_ATTRS ~WorkQueue();
  RT_API_ATTRS Terminator &terminator() { return terminator_; };

  // APIs for particular tasks.  These can return StatOk if the work is
  // completed immediately.
#ifdef RT_DEVICE_COMPILATION
  RT_API_ATTRS int BeginInitialize(const Descriptor &descriptor,
      const typeInfo::DerivedType &derived,
      MemcpyFct memcpyFct = &MemcpyWrapper) {
#else
  RT_API_ATTRS int BeginInitialize(const Descriptor &descriptor,
      const typeInfo::DerivedType &derived,
      MemcpyFct memcpyFct = &Fortran::runtime::memcpy) {
#endif
    if (runTicketsImmediately_) {
      return InitializeTicket{descriptor, derived, memcpyFct}.Run(*this);
    } else {
      StartTicket().emplace<InitializeTicket>(descriptor, derived, memcpyFct);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginInitializeClone(const Descriptor &clone,
      const Descriptor &original, const typeInfo::DerivedType &derived,
      bool hasStat, const Descriptor *errMsg) {
    if (runTicketsImmediately_) {
      return InitializeCloneTicket{clone, original, derived, hasStat, errMsg}
          .Run(*this);
    } else {
      StartTicket().emplace<InitializeCloneTicket>(
          clone, original, derived, hasStat, errMsg);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginFinalize(
      const Descriptor &descriptor, const typeInfo::DerivedType &derived) {
    if (runTicketsImmediately_) {
      return FinalizeTicket{descriptor, derived}.Run(*this);
    } else {
      StartTicket().emplace<FinalizeTicket>(descriptor, derived);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginDestroy(const Descriptor &descriptor,
      const typeInfo::DerivedType &derived, bool finalize) {
    if (runTicketsImmediately_) {
      return DestroyTicket{descriptor, derived, finalize}.Run(*this);
    } else {
      StartTicket().emplace<DestroyTicket>(descriptor, derived, finalize);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginAssign(Descriptor &to, const Descriptor &from,
      int flags, MemmoveFct memmoveFct,
      const typeInfo::DerivedType *declaredType) {
    if (runTicketsImmediately_) {
      return AssignTicket{to, from, flags, memmoveFct, declaredType}.Run(*this);
    } else {
      StartTicket().emplace<AssignTicket>(
          to, from, flags, memmoveFct, declaredType);
      return StatContinue;
    }
  }
  RT_API_ATTRS int BeginDerivedAssign(Descriptor &to, const Descriptor &from,
      const typeInfo::DerivedType &derived, int flags, MemmoveFct memmoveFct,
      Descriptor *deallocateAfter, bool isComponentwise) {
    if (runTicketsImmediately_) {
      return DerivedAssignTicket{
          to, from, derived, flags, memmoveFct, deallocateAfter, isComponentwise}
          .Run(*this);
    } else {
      StartTicket().emplace<DerivedAssignTicket>(
          to, from, derived, flags, memmoveFct, deallocateAfter, isComponentwise);
      return StatContinue;
    }
  }
  // IO ticket methods are only available on host - not used in GPU offloading
#if !defined(RT_DEVICE_COMPILATION)
  template <io::Direction DIR>
  RT_API_ATTRS int BeginDescriptorIo(io::IoStatementState &io,
      const Descriptor &descriptor, const io::NonTbpDefinedIoTable *table,
      bool &anyIoTookPlace) {
    if (runTicketsImmediately_) {
      return io::descr::DescriptorIoTicket<DIR>{
          io, descriptor, table, anyIoTookPlace}
          .Run(*this);
    } else {
      StartTicket().emplace<io::descr::DescriptorIoTicket<DIR>>(
          io, descriptor, table, anyIoTookPlace);
      return StatContinue;
    }
  }
  template <io::Direction DIR>
  RT_API_ATTRS int BeginDerivedIo(io::IoStatementState &io,
      const Descriptor &descriptor, const typeInfo::DerivedType &derived,
      const io::NonTbpDefinedIoTable *table, bool &anyIoTookPlace) {
    if (runTicketsImmediately_) {
      return io::descr::DerivedIoTicket<DIR>{
          io, descriptor, derived, table, anyIoTookPlace}
          .Run(*this);
    } else {
      StartTicket().emplace<io::descr::DerivedIoTicket<DIR>>(
          io, descriptor, derived, table, anyIoTookPlace);
      return StatContinue;
    }
  }
#endif

  RT_API_ATTRS int Run();

private:
#if RT_DEVICE_COMPILATION
  // Always use the work queue on a GPU device to avoid recursion.
  static constexpr bool runTicketsImmediately_{false};
#else
  // Avoid the work queue overhead on the host, unless it needs
  // debugging, which is so much easier there.
  static constexpr bool runTicketsImmediately_{true};
#endif

  // Most uses of the work queue won't go very deep.
  static constexpr int numStatic_{2};

  struct TicketList {
    bool isStatic{true};
    Ticket ticket;
    TicketList *previous{nullptr}, *next{nullptr};
  };

  RT_API_ATTRS Ticket &StartTicket();
  RT_API_ATTRS void Stop();

  Terminator &terminator_;
  TicketList *first_{nullptr}, *last_{nullptr}, *insertAfter_{nullptr};
  TicketList static_[numStatic_];
  TicketList *firstFree_{static_};
  bool anyDynamicAllocation_{false};
};

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_WORK_QUEUE_H_
