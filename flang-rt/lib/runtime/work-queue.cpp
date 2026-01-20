//===-- lib/runtime/work-queue.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/work-queue.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/type-info.h"

namespace Fortran::runtime {

#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
// FLANG_RT_DEBUG code is disabled when false.
static constexpr bool enableDebugOutput{false};
#endif

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS Ticket::Ticket() : begun{false}, type_{TicketType::Null} {}

RT_API_ATTRS Ticket::~Ticket() { destroy(); }

RT_API_ATTRS void Ticket::destroy() {
  switch (type_) {
  case TicketType::Null:
    // NullTicket has trivial destructor, nothing to do
    break;
  case TicketType::Initialize:
    reinterpret_cast<InitializeTicket *>(storage_.GetPtr())->~InitializeTicket();
    break;
  case TicketType::InitializeClone:
    reinterpret_cast<InitializeCloneTicket *>(storage_.GetPtr())
        ->~InitializeCloneTicket();
    break;
  case TicketType::Finalize:
    reinterpret_cast<FinalizeTicket *>(storage_.GetPtr())->~FinalizeTicket();
    break;
  case TicketType::Destroy:
    reinterpret_cast<DestroyTicket *>(storage_.GetPtr())->~DestroyTicket();
    break;
  case TicketType::Assign:
    reinterpret_cast<AssignTicket *>(storage_.GetPtr())->~AssignTicket();
    break;
  case TicketType::DerivedAssignFalse: {
    using TicketType = DerivedAssignTicket<false>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
  case TicketType::DerivedAssignTrue: {
    using TicketType = DerivedAssignTicket<true>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
#if !defined(RT_DEVICE_COMPILATION)
  case TicketType::DescriptorIoOutput: {
    using TicketType = io::descr::DescriptorIoTicket<io::Direction::Output>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
  case TicketType::DescriptorIoInput: {
    using TicketType = io::descr::DescriptorIoTicket<io::Direction::Input>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
  case TicketType::DerivedIoOutput: {
    using TicketType = io::descr::DerivedIoTicket<io::Direction::Output>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
  case TicketType::DerivedIoInput: {
    using TicketType = io::descr::DerivedIoTicket<io::Direction::Input>;
    reinterpret_cast<TicketType *>(storage_.GetPtr())->~TicketType();
    break;
  }
#endif
  }
  type_ = TicketType::Null;
}

RT_API_ATTRS int Ticket::dispatchBegin(WorkQueue &workQueue) {
  switch (type_) {
  case TicketType::Null:
    return reinterpret_cast<const NullTicket *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::Initialize:
    return reinterpret_cast<InitializeTicket *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::InitializeClone:
    return reinterpret_cast<InitializeCloneTicket *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::Finalize:
    return reinterpret_cast<FinalizeTicket *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::Destroy:
    return reinterpret_cast<DestroyTicket *>(storage_.GetPtr())->Begin(workQueue);
  case TicketType::Assign:
    return reinterpret_cast<AssignTicket *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::DerivedAssignFalse:
    return reinterpret_cast<DerivedAssignTicket<false> *>(storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::DerivedAssignTrue:
    return reinterpret_cast<DerivedAssignTicket<true> *>(storage_.GetPtr())
        ->Begin(workQueue);
#if !defined(RT_DEVICE_COMPILATION)
  case TicketType::DescriptorIoOutput:
    return reinterpret_cast<
               io::descr::DescriptorIoTicket<io::Direction::Output> *>(
               storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::DescriptorIoInput:
    return reinterpret_cast<
               io::descr::DescriptorIoTicket<io::Direction::Input> *>(
               storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::DerivedIoOutput:
    return reinterpret_cast<
               io::descr::DerivedIoTicket<io::Direction::Output> *>(
               storage_.GetPtr())
        ->Begin(workQueue);
  case TicketType::DerivedIoInput:
    return reinterpret_cast<
               io::descr::DerivedIoTicket<io::Direction::Input> *>(
               storage_.GetPtr())
        ->Begin(workQueue);
#endif
  }
  return StatOk; // Should never reach here
}

RT_API_ATTRS int Ticket::dispatchContinue(WorkQueue &workQueue) {
  switch (type_) {
  case TicketType::Null:
    return reinterpret_cast<const NullTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::Initialize:
    return reinterpret_cast<InitializeTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::InitializeClone:
    return reinterpret_cast<InitializeCloneTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::Finalize:
    return reinterpret_cast<FinalizeTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::Destroy:
    return reinterpret_cast<DestroyTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::Assign:
    return reinterpret_cast<AssignTicket *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::DerivedAssignFalse:
    return reinterpret_cast<DerivedAssignTicket<false> *>(storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::DerivedAssignTrue:
    return reinterpret_cast<DerivedAssignTicket<true> *>(storage_.GetPtr())
        ->Continue(workQueue);
#if !defined(RT_DEVICE_COMPILATION)
  case TicketType::DescriptorIoOutput:
    return reinterpret_cast<
               io::descr::DescriptorIoTicket<io::Direction::Output> *>(
               storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::DescriptorIoInput:
    return reinterpret_cast<
               io::descr::DescriptorIoTicket<io::Direction::Input> *>(
               storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::DerivedIoOutput:
    return reinterpret_cast<
               io::descr::DerivedIoTicket<io::Direction::Output> *>(
               storage_.GetPtr())
        ->Continue(workQueue);
  case TicketType::DerivedIoInput:
    return reinterpret_cast<
               io::descr::DerivedIoTicket<io::Direction::Input> *>(
               storage_.GetPtr())
        ->Continue(workQueue);
#endif
  }
  return StatOk; // Should never reach here
}

RT_API_ATTRS int Ticket::Continue(WorkQueue &workQueue) {
  if (!begun) {
    begun = true;
    return dispatchBegin(workQueue);
  } else {
    return dispatchContinue(workQueue);
  }
}

RT_API_ATTRS WorkQueue::~WorkQueue() {
  // Note: Ticket destructors will be called automatically when TicketList
  // objects are destroyed. The Ticket destructor will properly destroy the
  // stored ticket object via destroy().
  if (anyDynamicAllocation_) {
    if (last_) {
      if ((last_->next = firstFree_)) {
        last_->next->previous = last_;
      }
      firstFree_ = first_;
      first_ = last_ = nullptr;
    }
    while (firstFree_) {
      TicketList *next{firstFree_->next};
      if (!firstFree_->isStatic) {
        FreeMemory(firstFree_);
      }
      firstFree_ = next;
    }
  }
}

RT_API_ATTRS Ticket &WorkQueue::StartTicket() {
  if (!firstFree_) {
    void *p{AllocateMemoryOrCrash(terminator_, sizeof(TicketList))};
    firstFree_ = new (p) TicketList;
    firstFree_->isStatic = false;
    anyDynamicAllocation_ = true;
  }
  TicketList *newTicket{firstFree_};
  if ((firstFree_ = newTicket->next)) {
    firstFree_->previous = nullptr;
  }
  TicketList *after{insertAfter_ ? insertAfter_->next : nullptr};
  if ((newTicket->previous = insertAfter_ ? insertAfter_ : last_)) {
    newTicket->previous->next = newTicket;
  } else {
    first_ = newTicket;
  }
  if ((newTicket->next = after)) {
    after->previous = newTicket;
  } else {
    last_ = newTicket;
  }
  newTicket->ticket.begun = false;
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
  if (enableDebugOutput &&
      (executionEnvironment.internalDebugging &
          ExecutionEnvironment::WorkQueue)) {
    std::fprintf(stderr, "WQ: new ticket\n");
  }
#endif
  return newTicket->ticket;
}

RT_API_ATTRS int WorkQueue::Run() {
  while (last_) {
    TicketList *at{last_};
    insertAfter_ = last_;
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
    if (enableDebugOutput &&
        (executionEnvironment.internalDebugging &
            ExecutionEnvironment::WorkQueue)) {
      std::fprintf(stderr, "WQ: %zd %s\n", at->ticket.index(),
          at->ticket.begun ? "Continue" : "Begin");
    }
#endif
    int stat{at->ticket.Continue(*this)};
#if !defined(RT_DEVICE_COMPILATION) && !defined(OMP_OFFLOAD_BUILD)
    if (enableDebugOutput &&
        (executionEnvironment.internalDebugging &
            ExecutionEnvironment::WorkQueue)) {
      std::fprintf(stderr, "WQ: ... stat %d\n", stat);
    }
#endif
    insertAfter_ = nullptr;
    if (stat == StatOk) {
      if (at->previous) {
        at->previous->next = at->next;
      } else {
        first_ = at->next;
      }
      if (at->next) {
        at->next->previous = at->previous;
      } else {
        last_ = at->previous;
      }
      if ((at->next = firstFree_)) {
        at->next->previous = at;
      }
      at->previous = nullptr;
      firstFree_ = at;
    } else if (stat != StatContinue) {
      Stop();
      return stat;
    }
  }
  return StatOk;
}

RT_API_ATTRS void WorkQueue::Stop() {
  if (last_) {
    if ((last_->next = firstFree_)) {
      last_->next->previous = last_;
    }
    firstFree_ = first_;
    first_ = last_ = nullptr;
  }
}

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime
