//===-- asan_errors.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan implementation for error structures.
//===----------------------------------------------------------------------===//

#include "asan_errors.h"

#include "asan_descriptions.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __asan {

static void OnStackUnwind(const SignalContext &sig,
                          const void *callback_context,
                          BufferedStackTrace *stack) {
  bool fast = common_flags()->fast_unwind_on_fatal;
#if SANITIZER_FREEBSD || SANITIZER_NETBSD
  // On FreeBSD the slow unwinding that leverages _Unwind_Backtrace()
  // yields the call stack of the signal's handler and not of the code
  // that raised the signal (as it does on Linux).
  fast = true;
#endif
  // Tests and maybe some users expect that scariness is going to be printed
  // just before the stack. As only asan has scariness score we have no
  // corresponding code in the sanitizer_common and we use this callback to
  // print it.
  static_cast<const ScarinessScoreBase *>(callback_context)->Print();
  stack->Unwind(StackTrace::GetNextInstructionPc(sig.pc), sig.bp, sig.context,
                fast);
}

void ErrorDeadlySignal::Print() {
  ReportDeadlySignal(signal, tid, &OnStackUnwind, &scariness);
}

void ErrorDoubleFree::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: attempting %s on %p in thread %s:\n",
         scariness.GetDescription(), (void *)addr_description.addr,
         AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  scariness.Print();
  GET_STACK_TRACE_FATAL(second_free_stack->trace[0],
                        second_free_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), &stack);
}

void ErrorNewDeleteTypeMismatch::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s on %p in thread %s:\n",
         scariness.GetDescription(), (void *)addr_description.addr,
         AsanThreadIdAndName(tid).c_str());
  Printf("%s  object passed to delete has wrong type:\n", d.Default());
  if (delete_size != 0) {
    Printf(
        "  size of the allocated type:   %zd bytes;\n"
        "  size of the deallocated type: %zd bytes.\n",
        addr_description.chunk_access.chunk_size, delete_size);
  }
  const uptr user_alignment =
      addr_description.chunk_access.user_requested_alignment;
  if (delete_alignment != user_alignment) {
    char user_alignment_str[32];
    char delete_alignment_str[32];
    internal_snprintf(user_alignment_str, sizeof(user_alignment_str),
                      "%zd bytes", user_alignment);
    internal_snprintf(delete_alignment_str, sizeof(delete_alignment_str),
                      "%zd bytes", delete_alignment);
    static const char *kDefaultAlignment = "default-aligned";
    Printf(
        "  alignment of the allocated type:   %s;\n"
        "  alignment of the deallocated type: %s.\n",
        user_alignment > 0 ? user_alignment_str : kDefaultAlignment,
        delete_alignment > 0 ? delete_alignment_str : kDefaultAlignment);
  }
  CHECK_GT(free_stack->size, 0);
  scariness.Print();
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), &stack);
  Report(
      "HINT: if you don't care about these errors you may set "
      "ASAN_OPTIONS=new_delete_type_mismatch=0\n");
}

void ErrorFreeNotMalloced::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: attempting free on address "
      "which was not malloc()-ed: %p in thread %s\n",
      (void *)addr_description.Address(), AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  CHECK_GT(free_stack->size, 0);
  scariness.Print();
  GET_STACK_TRACE_FATAL(free_stack->trace[0], free_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), &stack);
}

void ErrorAllocTypeMismatch::Print() {
  static const char *alloc_names[] = {"INVALID", "malloc", "operator new",
                                      "operator new []"};
  static const char *dealloc_names[] = {"INVALID", "free", "operator delete",
                                        "operator delete []"};
  CHECK_NE(alloc_type, dealloc_type);
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s (%s vs %s) on %p\n",
         scariness.GetDescription(), alloc_names[alloc_type],
         dealloc_names[dealloc_type], (void *)addr_description.Address());
  Printf("%s", d.Default());
  CHECK_GT(dealloc_stack->size, 0);
  scariness.Print();
  GET_STACK_TRACE_FATAL(dealloc_stack->trace[0], dealloc_stack->top_frame_bp);
  stack.Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), &stack);
  Report(
      "HINT: if you don't care about these errors you may set "
      "ASAN_OPTIONS=alloc_dealloc_mismatch=0\n");
}

void ErrorMallocUsableSizeNotOwned::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: attempting to call malloc_usable_size() for "
      "pointer which is not owned: %p\n",
      (void *)addr_description.Address());
  Printf("%s", d.Default());
  stack->Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorSanitizerGetAllocatedSizeNotOwned::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: attempting to call "
      "__sanitizer_get_allocated_size() for pointer which is not owned: %p\n",
      (void *)addr_description.Address());
  Printf("%s", d.Default());
  stack->Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorCallocOverflow::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: calloc parameters overflow: count * size "
      "(%zd * %zd) cannot be represented in type size_t (thread %s)\n",
      count, size, AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorReallocArrayOverflow::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: reallocarray parameters overflow: count * size "
      "(%zd * %zd) cannot be represented in type size_t (thread %s)\n",
      count, size, AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorPvallocOverflow::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: pvalloc parameters overflow: size 0x%zx "
      "rounded up to system page size 0x%zx cannot be represented in type "
      "size_t (thread %s)\n",
      size, GetPageSizeCached(), AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorInvalidAllocationAlignment::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: invalid allocation alignment: %zd, "
      "alignment must be a power of two (thread %s)\n",
      alignment, AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorInvalidAlignedAllocAlignment::Print() {
  Decorator d;
  Printf("%s", d.Error());
#if SANITIZER_POSIX
  Report("ERROR: AddressSanitizer: invalid alignment requested in "
         "aligned_alloc: %zd, alignment must be a power of two and the "
         "requested size 0x%zx must be a multiple of alignment "
         "(thread %s)\n", alignment, size, AsanThreadIdAndName(tid).c_str());
#else
  Report("ERROR: AddressSanitizer: invalid alignment requested in "
         "aligned_alloc: %zd, the requested size 0x%zx must be a multiple of "
         "alignment (thread %s)\n", alignment, size,
         AsanThreadIdAndName(tid).c_str());
#endif
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorInvalidPosixMemalignAlignment::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: invalid alignment requested in posix_memalign: "
      "%zd, alignment must be a power of two and a multiple of sizeof(void*) "
      "== %zd (thread %s)\n",
      alignment, sizeof(void *), AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorAllocationSizeTooBig::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: requested allocation size 0x%zx (0x%zx after "
      "adjustments for alignment, red zones etc.) exceeds maximum supported "
      "size of 0x%zx (thread %s)\n",
      user_size, total_size, max_size, AsanThreadIdAndName(tid).c_str());
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorRssLimitExceeded::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: specified RSS limit exceeded, currently set to "
      "soft_rss_limit_mb=%zd\n", common_flags()->soft_rss_limit_mb);
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorOutOfMemory::Print() {
  Decorator d;
  Printf("%s", d.Error());
  ERROR_OOM("allocator is trying to allocate 0x%zx bytes\n", requested_size);
  Printf("%s", d.Default());
  stack->Print();
  PrintHintAllocatorCannotReturnNull();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorStringFunctionMemoryRangesOverlap::Print() {
  Decorator d;
  char bug_type[100];
  internal_snprintf(bug_type, sizeof(bug_type), "%s-param-overlap", function);
  Printf("%s", d.Error());
  Report(
      "ERROR: AddressSanitizer: %s: memory ranges [%p,%p) and [%p, %p) "
      "overlap\n",
      bug_type, (void *)addr1_description.Address(),
      (void *)(addr1_description.Address() + length1),
      (void *)addr2_description.Address(),
      (void *)(addr2_description.Address() + length2));
  Printf("%s", d.Default());
  scariness.Print();
  stack->Print();
  addr1_description.Print();
  addr2_description.Print();
  ReportErrorSummary(bug_type, stack);
}

void ErrorStringFunctionSizeOverflow::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s: (size=%zd)\n",
         scariness.GetDescription(), size);
  Printf("%s", d.Default());
  scariness.Print();
  stack->Print();
  addr_description.Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorBadParamsToAnnotateContiguousContainer::Print() {
  Report(
      "ERROR: AddressSanitizer: bad parameters to "
      "__sanitizer_annotate_contiguous_container:\n"
      "      beg     : %p\n"
      "      end     : %p\n"
      "      old_mid : %p\n"
      "      new_mid : %p\n",
      (void *)beg, (void *)end, (void *)old_mid, (void *)new_mid);
  stack->Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorBadParamsToAnnotateDoubleEndedContiguousContainer::Print() {
  Report(
      "ERROR: AddressSanitizer: bad parameters to "
      "__sanitizer_annotate_double_ended_contiguous_container:\n"
      "      storage_beg        : %p\n"
      "      storage_end        : %p\n"
      "      old_container_beg  : %p\n"
      "      old_container_end  : %p\n"
      "      new_container_beg  : %p\n"
      "      new_container_end  : %p\n",
      (void *)storage_beg, (void *)storage_end, (void *)old_container_beg,
      (void *)old_container_end, (void *)new_container_beg,
      (void *)new_container_end);
  stack->Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorBadParamsToCopyContiguousContainerAnnotations::Print() {
  Report(
      "ERROR: AddressSanitizer: bad parameters to "
      "__sanitizer_copy_contiguous_container_annotations:\n"
      "      src_storage_beg : %p\n"
      "      src_storage_end : %p\n"
      "      dst_storage_beg : %p\n"
      "      new_storage_end : %p\n",
      (void *)old_storage_beg, (void *)old_storage_end, (void *)new_storage_beg,
      (void *)new_storage_end);
  stack->Print();
  ReportErrorSummary(scariness.GetDescription(), stack);
}

void ErrorODRViolation::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s (%p):\n", scariness.GetDescription(),
         (void *)global1.beg);
  Printf("%s", d.Default());
  InternalScopedString g1_loc;
  InternalScopedString g2_loc;
  PrintGlobalLocation(&g1_loc, global1, /*print_module_name=*/true);
  PrintGlobalLocation(&g2_loc, global2, /*print_module_name=*/true);
  Printf("  [1] size=%zd '%s' %s\n", global1.size,
         MaybeDemangleGlobalName(global1.name), g1_loc.data());
  Printf("  [2] size=%zd '%s' %s\n", global2.size,
         MaybeDemangleGlobalName(global2.name), g2_loc.data());
  if (stack_id1 && stack_id2) {
    Printf("These globals were registered at these points:\n");
    Printf("  [1]:\n");
    StackDepotGet(stack_id1).Print();
    Printf("  [2]:\n");
    StackDepotGet(stack_id2).Print();
  }
  Report(
      "HINT: if you don't care about these errors you may set "
      "ASAN_OPTIONS=detect_odr_violation=0\n");
  InternalScopedString error_msg;
  error_msg.AppendF("%s: global '%s' at %s", scariness.GetDescription(),
                    MaybeDemangleGlobalName(global1.name), g1_loc.data());
  ReportErrorSummary(error_msg.data());
}

void ErrorInvalidPointerPair::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s: %p %p\n", scariness.GetDescription(),
         (void *)addr1_description.Address(),
         (void *)addr2_description.Address());
  Printf("%s", d.Default());
  GET_STACK_TRACE_FATAL(pc, bp);
  stack.Print();
  addr1_description.Print();
  addr2_description.Print();
  ReportErrorSummary(scariness.GetDescription(), &stack);
}

static bool AdjacentShadowValuesAreFullyPoisoned(u8 *s) {
  return s[-1] > 127 && s[1] > 127;
}

ErrorGenericBase::ErrorGenericBase(u32 tid, uptr addr, bool is_write_,
                                   uptr access_size_)
    : ErrorBase(tid),
      addr_description(addr, access_size_, /*shouldLockThreadRegistry=*/false),
      access_size(access_size_),
      is_write(is_write_),
      shadow_val(0) {
  scariness.Clear();
  if (access_size) {
    if (access_size <= 9) {
      char desr[] = "?-byte";
      desr[0] = '0' + access_size;
      scariness.Scare(access_size + access_size / 2, desr);
    } else if (access_size >= 10) {
      scariness.Scare(15, "multi-byte");
    }
    is_write ? scariness.Scare(20, "write") : scariness.Scare(1, "read");

    // Determine the error type.
    bug_descr = "unknown-crash";
    if (AddrIsInMem(addr)) {
      u8 *shadow_addr = (u8 *)MemToShadow(addr);
      // If we are accessing 16 bytes, look at the second shadow byte.
      if (*shadow_addr == 0 && access_size > ASAN_SHADOW_GRANULARITY)
        shadow_addr++;
      // If we are in the partial right redzone, look at the next shadow byte.
      if (*shadow_addr > 0 && *shadow_addr < 128) shadow_addr++;
      bool far_from_bounds = false;
      shadow_val = *shadow_addr;
      int bug_type_score = 0;
      // For use-after-frees reads are almost as bad as writes.
      int read_after_free_bonus = 0;
      switch (shadow_val) {
        case kAsanHeapLeftRedzoneMagic:
        case kAsanArrayCookieMagic:
          bug_descr = "heap-buffer-overflow";
          bug_type_score = 10;
          far_from_bounds = AdjacentShadowValuesAreFullyPoisoned(shadow_addr);
          break;
        case kAsanHeapFreeMagic:
          bug_descr = "heap-use-after-free";
          bug_type_score = 20;
          if (!is_write) read_after_free_bonus = 18;
          break;
        case kAsanStackLeftRedzoneMagic:
          bug_descr = "stack-buffer-underflow";
          bug_type_score = 25;
          far_from_bounds = AdjacentShadowValuesAreFullyPoisoned(shadow_addr);
          break;
        case kAsanInitializationOrderMagic:
          bug_descr = "initialization-order-fiasco";
          bug_type_score = 1;
          break;
        case kAsanStackMidRedzoneMagic:
        case kAsanStackRightRedzoneMagic:
          bug_descr = "stack-buffer-overflow";
          bug_type_score = 25;
          far_from_bounds = AdjacentShadowValuesAreFullyPoisoned(shadow_addr);
          break;
        case kAsanStackAfterReturnMagic:
          bug_descr = "stack-use-after-return";
          bug_type_score = 30;
          if (!is_write) read_after_free_bonus = 18;
          break;
        case kAsanUserPoisonedMemoryMagic:
          bug_descr = "use-after-poison";
          bug_type_score = 20;
          break;
        case kAsanContiguousContainerOOBMagic:
          bug_descr = "container-overflow";
          bug_type_score = 10;
          break;
        case kAsanStackUseAfterScopeMagic:
          bug_descr = "stack-use-after-scope";
          bug_type_score = 10;
          break;
        case kAsanGlobalRedzoneMagic:
          bug_descr = "global-buffer-overflow";
          bug_type_score = 10;
          far_from_bounds = AdjacentShadowValuesAreFullyPoisoned(shadow_addr);
          break;
        case kAsanIntraObjectRedzone:
          bug_descr = "intra-object-overflow";
          bug_type_score = 10;
          break;
        case kAsanAllocaLeftMagic:
        case kAsanAllocaRightMagic:
          bug_descr = "dynamic-stack-buffer-overflow";
          bug_type_score = 25;
          far_from_bounds = AdjacentShadowValuesAreFullyPoisoned(shadow_addr);
          break;
      }
      scariness.Scare(bug_type_score + read_after_free_bonus, bug_descr);
      if (far_from_bounds) scariness.Scare(10, "far-from-bounds");
    }
  }
}

ErrorGeneric::ErrorGeneric(u32 tid, uptr pc_, uptr bp_, uptr sp_, uptr addr,
                           bool is_write_, uptr access_size_)
    : ErrorGenericBase(tid, addr, is_write_, access_size_),
      pc(pc_),
      bp(bp_),
      sp(sp_) {}

static void PrintContainerOverflowHint() {
  Printf("HINT: if you don't care about these errors you may set "
         "ASAN_OPTIONS=detect_container_overflow=0.\n"
         "If you suspect a false positive see also: "
         "https://github.com/google/sanitizers/wiki/"
         "AddressSanitizerContainerOverflow.\n");
}

static void PrintShadowByte(InternalScopedString *str, const char *before,
    u8 byte, const char *after = "\n") {
  PrintMemoryByte(str, before, byte, /*in_shadow*/true, after);
}

static void PrintLegend(InternalScopedString *str) {
  str->AppendF(
      "Shadow byte legend (one shadow byte represents %d "
      "application bytes):\n",
      (int)ASAN_SHADOW_GRANULARITY);
  PrintShadowByte(str, "  Addressable:           ", 0);
  str->AppendF("  Partially addressable: ");
  for (u8 i = 1; i < ASAN_SHADOW_GRANULARITY; i++)
    PrintShadowByte(str, "", i, " ");
  str->AppendF("\n");
  PrintShadowByte(str, "  Heap left redzone:       ",
                  kAsanHeapLeftRedzoneMagic);
  PrintShadowByte(str, "  Freed heap region:       ", kAsanHeapFreeMagic);
  PrintShadowByte(str, "  Stack left redzone:      ",
                  kAsanStackLeftRedzoneMagic);
  PrintShadowByte(str, "  Stack mid redzone:       ",
                  kAsanStackMidRedzoneMagic);
  PrintShadowByte(str, "  Stack right redzone:     ",
                  kAsanStackRightRedzoneMagic);
  PrintShadowByte(str, "  Stack after return:      ",
                  kAsanStackAfterReturnMagic);
  PrintShadowByte(str, "  Stack use after scope:   ",
                  kAsanStackUseAfterScopeMagic);
  PrintShadowByte(str, "  Global redzone:          ", kAsanGlobalRedzoneMagic);
  PrintShadowByte(str, "  Global init order:       ",
                  kAsanInitializationOrderMagic);
  PrintShadowByte(str, "  Poisoned by user:        ",
                  kAsanUserPoisonedMemoryMagic);
  PrintShadowByte(str, "  Container overflow:      ",
                  kAsanContiguousContainerOOBMagic);
  PrintShadowByte(str, "  Array cookie:            ",
                  kAsanArrayCookieMagic);
  PrintShadowByte(str, "  Intra object redzone:    ",
                  kAsanIntraObjectRedzone);
  PrintShadowByte(str, "  ASan internal:           ", kAsanInternalHeapMagic);
  PrintShadowByte(str, "  Left alloca redzone:     ", kAsanAllocaLeftMagic);
  PrintShadowByte(str, "  Right alloca redzone:    ", kAsanAllocaRightMagic);
}

static void PrintShadowBytes(InternalScopedString *str, const char *before,
                             u8 *bytes, u8 *guilty, uptr n) {
  Decorator d;
  if (before)
    str->AppendF("%s%p:", before,
                 (void *)ShadowToMem(reinterpret_cast<uptr>(bytes)));
  for (uptr i = 0; i < n; i++) {
    u8 *p = bytes + i;
    const char *before =
        p == guilty ? "[" : (p - 1 == guilty && i != 0) ? "" : " ";
    const char *after = p == guilty ? "]" : "";
    PrintShadowByte(str, before, *p, after);
  }
  str->AppendF("\n");
}

static void PrintShadowMemoryForAddress(uptr addr) {
  if (!AddrIsInMem(addr)) return;
  uptr shadow_addr = MemToShadow(addr);
  const uptr n_bytes_per_row = 16;
  uptr aligned_shadow = shadow_addr & ~(n_bytes_per_row - 1);
  InternalScopedString str;
  str.AppendF("Shadow bytes around the buggy address:\n");
  for (int i = -5; i <= 5; i++) {
    uptr row_shadow_addr = aligned_shadow + i * n_bytes_per_row;
    // Skip rows that would be outside the shadow range. This can happen when
    // the user address is near the bottom, top, or shadow gap of the address
    // space.
    if (!AddrIsInShadow(row_shadow_addr)) continue;
    const char *prefix = (i == 0) ? "=>" : "  ";
    PrintShadowBytes(&str, prefix, (u8 *)row_shadow_addr, (u8 *)shadow_addr,
                     n_bytes_per_row);
  }
  if (flags()->print_legend) PrintLegend(&str);
  Printf("%s", str.data());
}

static void CheckPoisonRecords(uptr addr) {
  if (!AddrIsInMem(addr))
    return;

  u8 *shadow_addr = (u8 *)MemToShadow(addr);
  // If we are in the partial right redzone, look at the next shadow byte.
  if (*shadow_addr > 0 && *shadow_addr < 128)
    shadow_addr++;
  u8 shadow_val = *shadow_addr;

  if (shadow_val != kAsanUserPoisonedMemoryMagic)
    return;

  Printf("\n");

  if (flags()->poison_history_size <= 0) {
    Printf(
        "NOTE: the stack trace above identifies the code that *accessed* "
        "the poisoned memory.\n");
    Printf(
        "To identify the code that *poisoned* the memory, try the "
        "experimental setting ASAN_OPTIONS=poison_history_size=<size>.\n");
    return;
  }

  PoisonRecord record;
  if (FindPoisonRecord(addr, record)) {
    StackTrace poison_stack = StackDepotGet(record.stack_id);
    if (poison_stack.size > 0) {
      Printf("Memory was manually poisoned by thread T%u:\n", record.thread_id);
      poison_stack.Print();
    }
  } else {
    Printf("ERROR: no matching poison tracking record found.\n");
    Printf("Try a larger value for ASAN_OPTIONS=poison_history_size=<size>.\n");
  }
}

void ErrorGeneric::Print() {
  Decorator d;
  Printf("%s", d.Error());
  uptr addr = addr_description.Address();
  Report("ERROR: AddressSanitizer: %s on address %p at pc %p bp %p sp %p\n",
         bug_descr, (void *)addr, (void *)pc, (void *)bp, (void *)sp);
  Printf("%s", d.Default());

  Printf("%s%s of size %zu at %p thread %s%s\n", d.Access(),
         access_size ? (is_write ? "WRITE" : "READ") : "ACCESS", access_size,
         (void *)addr, AsanThreadIdAndName(tid).c_str(), d.Default());

  scariness.Print();
  GET_STACK_TRACE_FATAL(pc, bp);
  stack.Print();

  // Pass bug_descr because we have a special case for
  // initialization-order-fiasco
  addr_description.Print(bug_descr);
  if (shadow_val == kAsanContiguousContainerOOBMagic)
    PrintContainerOverflowHint();
  ReportErrorSummary(bug_descr, &stack);
  PrintShadowMemoryForAddress(addr);

  // This is an experimental flag, hence we don't make a special handler.
  CheckPoisonRecords(addr);
}

ErrorNonSelfGeneric::ErrorNonSelfGeneric(uptr *callstack_, u32 n_callstack,
                                         uptr *addrs, u32 n_addrs,
                                         u64 *threadids, u32 n_threads,
                                         bool is_write, u32 access_size,
                                         int fd_, s64 vm_adj, u64 off_, u64 sz_)
    : ErrorGenericBase(kInvalidTid, addrs[0], is_write, access_size),
      cb_loc(fd_, vm_adj, off_, sz_) {
  for (u64 i = 0; i < Min(addr_count, n_addrs); i++) addresses[i] = addrs[i];
  for (u64 i = 0; i < Min(threads_count, n_threads); i++)
    thread_id[i] = threadids[i];
  for (u64 i = 0; i < Min(maxcs_depth, n_callstack); i++)
    callstack[i] = callstack_[i];
}

void ErrorNonSelfGeneric::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s on address %p at pc %p\n", bug_descr,
         (void *)addresses[0], callstack[0]);

  Printf("%s%s of size %zu at %p thread id %zu\n", d.Access(),
         access_size ? (is_write ? "WRITE" : "READ") : "ACCESS", access_size,
         (void *)addresses[0], thread_id[0]);

  // todo: perform symbolization for the given callstack
  // can be done by creating in-memory object file or by writing
  // data to a temporary file or by findng the filepath by following
  // /proc/PID/fd
  Printf("%s", d.Default());
  Printf("AddressSanitizer cannot provide additional information!\n");
  PrintShadowMemoryForAddress(addresses[0]);
}

ErrorNonSelfAMDGPU::ErrorNonSelfAMDGPU(uptr *dev_callstack, u32 n_callstack,
                                       uptr *dev_address, u32 n_addrs,
                                       u64 *wi_ids, u32 n_wi, bool is_write_,
                                       u32 access_size_, int fd_, s64 vm_adj,
                                       u64 file_start_, u64 file_size_)
    : ErrorGenericBase(kInvalidTid, dev_address[0], is_write_, access_size_),
      cb_loc(fd_, vm_adj, file_start_, file_size_),
      wg(),
      nactive_threads(n_addrs),
      device_id(0) {
  if (nactive_threads > wavesize)
    nactive_threads = wavesize;

  callstack[0] = dev_callstack[0];
  device_id = wi_ids[0];
  wg.idx = wi_ids[1];
  wg.idy = wi_ids[2];
  wg.idz = wi_ids[3];
  wi_ids += 4;
  for (u64 i = 0; i < nactive_threads; i++) {
    device_address[i] = dev_address[i];
    workitem_ids[i] = wi_ids[i];
  }
}

void ErrorNonSelfAMDGPU::PrintStack() {
  InternalScopedString source_location;
  source_location.AppendF("  #0 %p", callstack[0]);
#if SANITIZER_AMDGPU
  source_location.Append(" in ");
  __sanitizer::AMDGPUCodeObjectSymbolizer symbolizer;
  symbolizer.Init(cb_loc.fd, cb_loc.offset, cb_loc.size);
  symbolizer.SymbolizePC(callstack[0] - cb_loc.vma_adjust, source_location);
  // release all allocated comgr objects.
  symbolizer.Release();
#endif
  Printf("%s", source_location.data());
}

void ErrorNonSelfAMDGPU::PrintThreadsAndAddresses() {
  InternalScopedString str;
  str.Append("Thread ids and accessed addresses:\n");
  for (u32 idx = 0, per_row_count = 0; idx < nactive_threads; idx++) {
    // print 8 threads per row.
    if (per_row_count == 8) {
      str.Append("\n");
      per_row_count = 0;
    }
    str.AppendF("%02d : %p ", workitem_ids[idx], device_address[idx]);
    per_row_count++;
  }
  str.Append("\n");
  Printf("%s\n", str.data());
}

static uptr ScanForMagicDown(uptr start, uptr lo, uptr magic0, uptr magic1) {
  for (uptr p = start; p >= lo; p -= sizeof(uptr)) {
    if (((uptr*)p)[0] == magic0 && ((uptr*)p)[1] == magic1)
      return p;
  }
  return 0;
}

static uptr ScanForMagicUp(uptr start, uptr hi, uptr magic0, uptr magic1) {
  for (uptr p = start; p < hi; p += sizeof(uptr)) {
    if (((uptr*)p)[0] == magic0 && ((uptr*)p)[1] == magic1)
      return p;
  }
  return 0;
}

void ErrorNonSelfAMDGPU::PrintMallocStack() {
  // Facts about asan malloc on device
  const uptr magic = static_cast<uptr>(0xfedcba1ee1abcdefULL);
  const uptr offset = 32;
  const uptr min_chunk_size = 96;
  const uptr min_alloc_size = 48;

  Decorator d;
  HeapAddressDescription addr_description;

  if (GetHeapAddressInformation(device_address[0], access_size,
              &addr_description) &&
      addr_description.chunk_access.chunk_size >= min_chunk_size) {
    uptr lo = addr_description.chunk_access.chunk_begin;
    uptr hi = lo + addr_description.chunk_access.chunk_size - min_alloc_size;
    uptr start = RoundDownTo(device_address[0], sizeof(uptr));

    uptr plo = ScanForMagicDown(start, lo, magic, lo);
    if (plo) {
      callstack[0] = ((uptr*)plo)[2];
      Printf("%s%p is %u bytes above an address from a %sdevice malloc "
              "(or free) call of size %u from%s\n",
              d.Location(), device_address[0],
              (int)(device_address[0] - (plo+offset)),
              d.Allocation(), ((int*)plo)[7], d.Default());
      // TODO: The code object with the malloc call may not be the same
      // code object trying the illegal access.  A mechanism is needed
      // to obtain the former.
      PrintStack();
    }

    uptr phi = ScanForMagicUp(start, hi, magic, lo);
    if (phi) {
      callstack[0] = ((uptr*)phi)[2];
      Printf("%s%p is %u bytes below an address from a %sdevice malloc "
              "(or free) call of size %u from%s\n",
              d.Location(), device_address[0],
              (int)((phi+offset) - device_address[0]),

              d.Allocation(), ((int*)phi)[7], d.Default());
      PrintStack();
    }
  }
}

void ErrorNonSelfAMDGPU::Print() {
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: AddressSanitizer: %s on amdgpu device %zu at pc %p\n",
         bug_descr, device_id, callstack[0]);
  Printf("%s%s of size %zu in workgroup id (%zu,%zu,%zu)\n", d.Access(),
         (is_write ? "WRITE" : "READ"), access_size, wg.idx, wg.idy, wg.idz);
  Printf("%s", d.Default());
  PrintStack();
  Printf("%s", d.Location());
  PrintThreadsAndAddresses();
  Printf("%s", d.Default());
  if (shadow_val == kAsanHeapFreeMagic ||
      shadow_val == kAsanHeapLeftRedzoneMagic ||
      shadow_val == kAsanArrayCookieMagic) {
    PrintMallocStack();
  }
  addr_description.Print(bug_descr, true);
  Printf("%s", d.Default());
  // print shadow memory region for single address
  PrintShadowMemoryForAddress(device_address[0]);
}
}  // namespace __asan
