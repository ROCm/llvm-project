//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is an extension of rpc_server.h
//
// Consumers must add the Clang resource header directory to the include path
// when compiling translation units that include this header (directly or via
// <shared/emissary_rpc_server.h>). EmissaryIds.h is installed there, not under
// lib/llvm/include:
//
//   -I$("$CXX" -print-resource-dir)/include
//
// Typical HIP/OpenMP demo builds also pass -I for lib/llvm/include (or
// llvm/include) so that <shared/emissary_rpc_server.h> resolves.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H

#if __has_include("../clang/lib/Headers/EmissaryIds.h")
#include "../clang/lib/Headers/EmissaryIds.h"
#else
#include "EmissaryIds.h"
#endif

#include "rpc.h"
#include "rpc_opcodes.h"
#include <EmissaryIds.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// Emissary host handler registry (component D1 of the multi-client design)
//
// Runtime registry that maps an Emissary API id to the host handler that
// services it. It lets a client library register its dispatcher at load time
// so the RPC server (EmissaryTop) can route requests without a compile-time
// switch over every known client.
//
// The registry is a fixed-size table with no dynamic allocation.
// EmissaryRegister and EmissaryLookup have C linkage so a client shared object
// can register against the server without C++ name-mangling coupling. The
// backing table is a C++17 inline variable, so all translation units in a
// program share one instance; on ELF it also merges across shared objects
// under default visibility, letting a client .so and the server share the same
// table.
//===----------------------------------------------------------------------===//

/// Upper bound on the number of distinct Emissary API ids. The table is indexed
/// directly by \c emisid, so this also bounds the largest id that can be
/// registered. It comfortably exceeds the current \c offload_emis_id_t range
/// and leaves room for reserved/dynamic ids.
#define EMISSARY_MAX_REGISTERED_IDS 64

/// Host handler for one Emissary API id. The signature matches the per-client
/// dispatchers (\c EmissaryMPI, \c EmissaryHDF5, ...): it receives the RPC data
/// buffer, the decoded argument descriptor, and the unpacked argument vector.
typedef EmissaryReturn_t (*EmissaryHandler_t)(char *data, emisArgBuf_t *ab,
                                              emis_argptr_t *args[]);

namespace emissary_registry_detail {
/// Backing table, indexed directly by Emissary API id. As a C++17 inline
/// variable it has exactly one instance across the whole program, but that
/// only holds across shared objects if the symbol keeps default visibility
/// *and* the linker is told to export it. Two things are needed to make that
/// true even when a consuming DSO is linked with an explicit
/// `--version-script` (as libomptarget/liboffload are) and/or
/// -fvisibility-inlines-hidden (LLVM's default project-wide flag), either of
/// which would otherwise localize this vague-linkage symbol into each DSO and
/// silently defeat the cross-DSO sharing this registry depends on:
///  1. Explicit default visibility, so the compiler doesn't hide it.
///  2. A stable `asm` symbol name, so it can be listed by name in a
///     `--version-script` `global:` clause without embedding an
///     Itanium-mangled C++ symbol (`_ZN...`) in linker input -- mangled names
///     are an ABI/compiler-version implementation detail, not something
///     version scripts should hardcode.
/// \internal
__attribute__((visibility("default")))
inline EmissaryHandler_t Table[EMISSARY_MAX_REGISTERED_IDS] asm(
    "EmissaryRegistryTable") = {};
} // namespace emissary_registry_detail

extern "C" {

/// Register a host handler for an Emissary API id.
///
/// \param emisid the Emissary API id (an \c offload_emis_id_t value or a
///   reserved dynamic id) to associate with \p handler.
/// \param handler the host dispatcher to invoke for \p emisid; must not be
///   null.
/// \returns \c true on success; \c false if \p emisid is out of range,
///   \p handler is null, or a *different* handler is already registered for
///   \p emisid. Re-registering the identical handler is idempotent and
///   succeeds.
///
/// Explicit default visibility (see \c Table above): this symbol must merge
/// across DSOs even when the caller is built with -fvisibility-inlines-hidden.
__attribute__((visibility("default")))
inline bool EmissaryRegister(unsigned int emisid, EmissaryHandler_t handler) {
  if (emisid >= EMISSARY_MAX_REGISTERED_IDS || handler == nullptr)
    return false;
  EmissaryHandler_t &Slot = emissary_registry_detail::Table[emisid];
  // Reject last-wins: two libraries claiming the same id is a configuration
  // error, not something to silently overwrite.
  if (Slot != nullptr && Slot != handler)
    return false;
  Slot = handler;
  return true;
}

/// Look up the host handler registered for an Emissary API id.
///
/// \param emisid the Emissary API id to look up.
/// \returns the registered handler, or null if \p emisid is out of range or no
///   handler is registered for it.
///
/// Explicit default visibility (see \c Table above): this symbol must merge
/// across DSOs even when the caller is built with -fvisibility-inlines-hidden.
__attribute__((visibility("default")))
inline EmissaryHandler_t EmissaryLookup(unsigned int emisid) {
  if (emisid >= EMISSARY_MAX_REGISTERED_IDS)
    return nullptr;
  return emissary_registry_detail::Table[emisid];
}

} // extern "C"

// No Emissary API host handler is declared or called here by name anymore.
// Every client -- MPI, HDF5, PRINT, and RESERVE -- self-registers its host
// dispatcher with the runtime registry defined above, so EmissaryTop routes all
// of them through EmissaryLookup without a compile-time weak symbol or switch
// case. PRINT is the toolchain-provided built-in; it self-registers from this
// header (see emissary_print_self_register below). RESERVE is registered by the
// user/reserved client library. EMIS_ID_FORTRT is a reserved wire-format id
// with no current provider: device Fortran I/O is serviced by flang-rt's own
// generic RPC path, not through Emissary. If a future Fortran runtime routes
// I/O through Emissary, it would self-register EMIS_ID_FORTRT the same way.
extern "C" {
/// Optional FORCE_OPT=1 SDMA path for device MPI Put/Get (libemissary_mpi).
/// This is an internal optimization hook, not an Emissary API, so it keeps its
/// weak-stub design: libLLVMOffload links without libemissary_mpi; the app
/// overrides with a strong definition from libemissary_mpi when FORCE_OPT SDMA
/// is used.
__attribute__((weak)) int emissary_mpi_sdma_try_dm_buffer(
    char *rpc_buffer, unsigned long long *out_result) {
  (void)rpc_buffer;
  (void)out_result;
  return -1;
}
} // end extern "C"

namespace rpc {
namespace internal {

// NUMFPREGS and FPREGSZ are part of x86 vargs ABI that
// is recreated with this printf support.
#define NUMFPREGS 8
#define FPREGSZ 16

typedef int uint128_t __attribute__((mode(TI)));
struct emissary_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct emissary_pfIntRegs emissary_pfIntRegs_t; // size = 48 bytes

struct emissary_pfRegSaveArea {
  emissary_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct emissary_pfRegSaveArea
    emissary_pfRegSaveArea_t; // size = 304 bytes

struct emissary_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  emissary_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  size_t overflow_size;
} __attribute__((packed));
typedef struct emissary_ValistExt emissary_ValistExt_t;

// Handle overflow when building the va_list for vprintf
static service_rc emissary_pfGetOverflow(emissary_ValistExt_t *valist,
                                         size_t needsize) {
  if (needsize < valist->overflow_size)
    return _ERC_SUCCESS;

  // Make the overflow area bigger
  size_t stacksize;
  void *newstack;
  if (valist->overflow_size == 0) {
    // Make initial save area big to reduce mallocs
    stacksize = (FPREGSZ * NUMFPREGS) * 2;
    if (needsize > stacksize)
      stacksize = needsize; // maybe a big string
  } else {
    // Initial save area not big enough, double it
    stacksize = valist->overflow_size * 2;
  }
  if (!(newstack = malloc(stacksize))) {
    return _ERC_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return _ERC_SUCCESS;
}

// Add an integer to the va_list for vprintf
static service_rc emissary_pfAddInteger(emissary_ValistExt_t *valist, char *val,
                                        size_t valsize, size_t *stacksize) {
  uint64_t ival;
  switch (valsize) {
  case 1:
    ival = *(uint8_t *)val;
    break;
  case 2:
    ival = *(uint32_t *)val;
    break;
  case 4:
    ival = (*(uint32_t *)val);
    break;
  case 8:
    ival = *(uint64_t *)val;
    break;
  default: {
    return _ERC_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes, sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return _ERC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return _ERC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static service_rc emissary_pfAddString(emissary_ValistExt_t *valist, char *val,
                                       size_t strsz, size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return _ERC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return _ERC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static service_rc emissary_pfAddFloat(emissary_ValistExt_t *valist,
                                      char *numdata, size_t valsize,
                                      size_t *stacksize) {
  // we could use load because doubles are now aligned
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    return _ERC_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(emissary_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return _ERC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return _ERC_SUCCESS;
}

// Build an extended va_list for vprintf by unpacking the buffer
static service_rc emissary_pfBuildValist(emissary_ValistExt_t *valist,
                                         int NumArgs, char *keyptr,
                                         char *dataptr, char *strptr,
                                         unsigned long long *data_not_used) {
  emissary_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (emissary_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return _ERC_STATUS_ERROR;
  memset(regs, 0, regs_size);
  *valist = (emissary_ValistExt_t){
      .gp_offset = 0,
      .fp_offset = 0,
      .overflow_arg_area = NULL,
      .reg_save_area = regs,
      .overflow_size = 0,
  };

  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  size_t stacksize = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int emisid = key >> 16;
    unsigned int numbits = (key << 16) >> 16;
    switch (emisid) {
    case EmisFloatTy:
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(emissary_pfIntRegs_t);
      if (emissary_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
        return _ERC_ADDFLOAT_ERROR;
      break;

    case EmisIntegerTy:
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;
      if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return _ERC_ADDINT_ERROR;
      break;

    case EmisPointerTy: {
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        if (strsz == 0) {
          if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
            return _ERC_ADDINT_ERROR;
        } else {
          if (emissary_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
            return _ERC_ADDSTRING_ERROR;
        }
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return _ERC_ADDINT_ERROR;
      }
    } break;
    default:
      return _ERC_INVALID_ID_ERROR;
    }

    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
  }
  return _ERC_SUCCESS;
} // end emissary_pfBuildValist

static service_rc emissary_printf(uint *rc, emisArgBuf_t *ab) {
  if (ab->DataLen == 0)
    return _ERC_SUCCESS;

  char *fmtstr = ab->strptr;

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t) * (unsigned int *)ab->argptr;

  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist; // FIXME: We may need to align this declare
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _ERC_SUCCESS)
    return _ERC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;
  *rc = vprintf(fmtstr, *real_va_list);
  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);
  return _ERC_SUCCESS;
}

// emisExtractArgBuf extract ArgBuf using protocol EmitEmissaryExec makes.
static void emisExtractArgBuf(char *data, emisArgBuf_t *ab) {

  uint32_t *int32_data = (uint32_t *)data;
  ab->DataLen = int32_data[0];
  ab->NumArgs = int32_data[1];

  // Note: while the data buffer contains all args including strings,
  // ab->DataLen does not include strings. It only counts header, keys,
  // and aligned numerics.

  ab->keyptr = data + (2 * sizeof(int));
  ab->argptr = ab->keyptr + (ab->NumArgs * sizeof(int));
  ab->strptr = data + (size_t)ab->DataLen;
  int alignfill = 0;
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    alignfill = 4;
  }

  // Extract the two emissary identifiers and number of send
  // and recv device data transfers. These are 4 16 bit values
  // packed into a single 64-bit field.
  uint64_t arg1 = *(uint64_t *)ab->argptr;
  ab->emisid = (unsigned int)((arg1 >> 48) & 0xFFFF);
  ab->emisfnid = (unsigned int)((arg1 >> 32) & 0xFFFF);
  ab->NumSendXfers = (unsigned int)((arg1 >> 16) & 0xFFFF);
  ab->NumRecvXfers = (unsigned int)((arg1) & 0xFFFF);

  // skip the uint64_t emissary id arg which is first arg in _emissary_exec.
  ab->keyptr += sizeof(int);
  ab->argptr += sizeof(uint64_t);
  ab->NumArgs -= 1;

  // data_not_used used for testing consistency.
  ab->data_not_used =
      (size_t)(ab->DataLen) - (((size_t)(3 + ab->NumArgs) * sizeof(int)) +
                               sizeof(uint64_t) + alignfill);

  // Ensure first arg after emissary id arg is aligned.
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    ab->data_not_used -= 4;
  }
}

/// Get uint32 value extended to uint64_t value from a char ptr
static uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}

/// Get uint64_t value from a char ptr
static uint64_t getuint64(char *val) { return *(uint64_t *)val; }

// build argument array to create call to variadic wrappers
static uint32_t
EmissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr, char *strptr,
                   unsigned long long *data_not_used, emis_argptr_t *a[],
                   std::unordered_map<void *, void *> *D2HAddrList) {
  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;
  uint argcount = 0;
  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int emis_id = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (emis_id) {
    case EmisFloatTy:
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      break;

    case EmisIntegerTy:
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      break;

    case EmisPointerTy: {
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)((char *)strptr);
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      }
      if (D2HAddrList) {
        auto found = D2HAddrList->find((void *)a[argcount]);
        if (found != D2HAddrList->end())
          a[argcount] = (emis_argptr_t *)found->second;
      }
    } break;

    default:
      return _ERC_INVALID_ID_ERROR;
    }
    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return _ERC_SUCCESS;
}

//  Utility to skip two args in the ArgBuf
static void emisSkipXferArgSet(emisArgBuf_t *ab) {
  // Skip the ptr and size of the Xfer
  ab->NumArgs -= 2;
  ab->keyptr += 2 * sizeof(uint32_t);
  ab->argptr += 2 * sizeof(void *);
  ab->data_not_used -= 2 * sizeof(void *);
}

static service_rc emissary_fprintf(uint *rc, emisArgBuf_t *ab) {

  if (ab->DataLen == 0)
    return _ERC_SUCCESS;
  char *fmtstr = ab->strptr;
  FILE *fileptr = (FILE *)*((size_t *)ab->argptr);

  // Skip past the file pointer
  ab->NumArgs--;
  ab->keyptr += 4;
  ab->argptr += sizeof(FILE *);
  ab->data_not_used -= sizeof(FILE *);

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t) * (unsigned int *)ab->argptr;
  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist; // FIXME: we may want to align this declare
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _ERC_SUCCESS)
    return _ERC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;
  *rc = vfprintf(fileptr, fmtstr, *real_va_list);
  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);
  return _ERC_SUCCESS;
}

// PRINT host dispatcher. It matches EmissaryHandler_t so it can be stored in
// the runtime registry like every other client, even though it services printf
// entirely from the raw argument buffer and does not consult the unpacked
// argument vector (args is intentionally unused).
static EmissaryReturn_t EmissaryPrint(char *data, emisArgBuf_t *ab,
                                      emis_argptr_t *args[]) {
  (void)data;
  (void)args;
  uint32_t return_value;
  service_rc rc;
  switch (ab->emisfnid) {
  case _printf_idx: {
    rc = emissary_printf(&return_value, ab);
    break;
  }
  case _fprintf_idx: {
    rc = emissary_fprintf(&return_value, ab);
    break;
  }
  case _ockl_asan_report_idx: {
    fprintf(stderr, " asan_report not yet implemented\n");
    return_value = 0;
    rc = _ERC_STATUS_ERROR;
    break;
  }
  case _print_INVALID:
  default: {
    fprintf(stderr, " INVALID emissary function id (%d) for PRINT API \n",
            ab->emisfnid);
    return_value = 0;
    rc = _ERC_STATUS_ERROR;
    break;
  }
  }
  if (rc != _ERC_SUCCESS)
    fprintf(stderr, "HOST failure in _emissary_execute_print rc:%d\n", rc);

  return (EmissaryReturn_t)return_value;
}

// Self-registration of the built-in PRINT service. Unlike MPI/HDF5/RESERVE,
// PRINT is provided by the toolchain rather than a separate client library, so
// it registers from this header: any RPC server translation unit that includes
// <emissary_rpc_server.h> (the offload runtime and the demo servers) gains
// printf/fprintf support automatically. The constructor is static so it does
// not collide across translation units, and EmissaryRegister is idempotent, so
// redundant registration from multiple includers is harmless.
namespace emissary_registry_detail {
__attribute__((constructor)) static void emissary_print_self_register(void) {
  EmissaryRegister(EMIS_ID_PRINT, &EmissaryPrint);
}
} // namespace emissary_registry_detail

static EmissaryReturn_t
EmissaryTop(char *data, emisArgBuf_t *ab,
            std::unordered_map<void *, void *> *D2HAddrList) {
  // Registry-only dispatch (D1): every Emissary API -- MPI, HDF5, PRINT,
  // RESERVE, and any out-of-tree client -- is serviced through the runtime
  // registry. A client's handler is present because its library (or, for the
  // built-in PRINT service, this header) self-registered at load time. There is
  // no per-client switch and no weak symbol fallback: an unregistered id (for
  // example the reserved-but-unused EMIS_ID_FORTRT) is simply unsupported.
  if (ab->emisid == EMIS_ID_INVALID) {
    fprintf(stderr, "Emissary (host execution) got invalid EMIS_ID\n");
    return (EmissaryReturn_t)0;
  }

  EmissaryHandler_t handler = EmissaryLookup(ab->emisid);
  if (handler == nullptr) {
    fprintf(stderr,
            "Emissary (host execution) EMIS_ID:%d fnid:%d not supported\n",
            ab->emisid, ab->emisfnid);
    return (EmissaryReturn_t)0;
  }

  emis_argptr_t **args = (emis_argptr_t **)aligned_alloc(
      sizeof(emis_argptr_t), ab->NumArgs * sizeof(emis_argptr_t *));

  // Build the unpacked argument vector against a scratch copy of data_not_used
  // so the buffer descriptor (ab) is left pristine for the handler. PRINT walks
  // the raw buffer itself and relies on ab->data_not_used being intact; the
  // other handlers use only the argument vector, so this is safe for all of
  // them and keeps a single uniform dispatch path.
  unsigned long long data_not_used = ab->data_not_used;
  if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                         &data_not_used, &args[0],
                         D2HAddrList) != _ERC_SUCCESS) {
    free(args);
    return (EmissaryReturn_t)0;
  }

  EmissaryReturn_t result = handler(data, ab, args);
  free(args);
  return result;
}

// -----------------------------------------------------------------
// -- Handle OFFLOAD_EMISSARY and OFFLOAD_EMISSARY_DM opcodes     --
// -- handle_emissary_impl calls EmissaryTop for each active lane --
// -----------------------------------------------------------------
template <uint32_t NumLanes>
inline RPCStatus handle_emissary_impl(Server::Port &port) {


  switch (port.get_opcode()) {

  // This case handles the device function __llvm_emissary_rpc for emissary
  // APIs that require no d2h or h2d memory transfer.
  case OFFLOAD_EMISSARY: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *buf_ptrs[NumLanes] = {nullptr};
    port.recv_n(buf_ptrs, Sizes, [&](uint64_t Size) { return new char[Size]; });
    uint32_t id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        emisArgBuf_t ab;
        emisExtractArgBuf((char *)buffer_ptr, &ab);
        Results[id++] = EmissaryTop((char *)buffer_ptr, &ab, nullptr);
      }
    }
    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
    });
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        delete[] reinterpret_cast<char *>(buffer_ptr);
      }
    }
    break;
  }

  // This case handles the device function __llvm_emissary_rpc_dm for emissary
  // APIs require D2H or H2D transfer vectors to be processed through the port.
  // FIXME: test with multiple transfer vectors of the same type.
  case OFFLOAD_EMISSARY_DM: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *buf_ptrs[NumLanes] = {nullptr};
    port.recv_n(buf_ptrs, Sizes, [&](uint64_t Size) { return new char[Size]; });

    uint32_t id = 0;
    emisArgBuf_t AB[NumLanes];
    std::unordered_map<void *, void *> D2HAddrList;
    void *Xfers[NumLanes] = {nullptr};
    void *devXfers[NumLanes] = {nullptr};
    uint64_t XferSzs[NumLanes] = {0};
    bool sdma_handled[NumLanes] = {false};
    uint32_t numSendXfers = 0;
    id = 0;

    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {

        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)buffer_ptr, ab);
        unsigned long long sdma_result = 0;
        if (emissary_mpi_sdma_try_dm_buffer((char *)buffer_ptr, &sdma_result) ==
            0) {
          Results[id] = sdma_result;
          sdma_handled[id] = true;
          id++;
          continue;
        }
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          numSendXfers++;
          devXfers[id] = (void *)*((uint64_t *)ab->argptr);
          XferSzs[id] = (size_t) * ((size_t *)(ab->argptr + sizeof(void *)));
          emisSkipXferArgSet(ab);
        }
        // Allocate the host space for the receive Xfers
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          size_t devSz =
              (((size_t) * ((size_t *)(ab->argptr + sizeof(void *)))) &
               0x00000000FFFFFFFF);
          void *hostAddr = new char[devSz];
          D2HAddrList.insert(std::pair<void *, void *>(devAddr, hostAddr));
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    // recv_n for device send_n into new host-allocated Xfers
    if (numSendXfers)
      port.recv_n(Xfers, XferSzs,
                  [&](uint64_t Size) { return new char[Size]; });

    // Xfers now contains just allocated host addrs for sends and
    // devXfers contains corresponding devAddr for those sends
    // Build map to pass to Emissary
    id = 0;
    for (void *Xfer : Xfers) {
      if (Xfer) {
        D2HAddrList.insert(std::pair<void *, void *>(devXfers[id], Xfer));
        id++;
      }
    }

    // Call EmissaryTop for each active lane
    id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        if (sdma_handled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)buffer_ptr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++)
          emisSkipXferArgSet(ab);
        Results[id] = EmissaryTop((char *)buffer_ptr, ab, &D2HAddrList);
        id++;
      }
    }

    // Process send_n for the H2D Xfers.
    void *recvXfers[NumLanes] = {nullptr};
    uint64_t recvXferSzs[NumLanes] = {0};
    id = 0;
    uint32_t numRecvXfers = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        if (sdma_handled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        // Reset ArgBuf tracker
        emisExtractArgBuf((char *)buffer_ptr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          numRecvXfers++;
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          recvXfers[id] = D2HAddrList[devAddr];
          recvXferSzs[id] =
              (((uint64_t) * ((size_t *)(ab->argptr + sizeof(void *)))) &
               0x00000000FFFFFFFF);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }
    if (numRecvXfers)
      port.send_n(recvXfers, recvXferSzs);

    // Cleanup all host allocated transfer buffers
    id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        if (sdma_handled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        // Reset the ArgBuf tracker ab
        emisExtractArgBuf((char *)buffer_ptr, ab);
        // Cleanup host allocated send Xfers
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          void *hostAddr = D2HAddrList[devAddr];
          delete[] reinterpret_cast<char *>(hostAddr);
          emisSkipXferArgSet(ab);
        }
        // Cleanup host allocated bufs
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          void *hostAddr = D2HAddrList[devAddr];
          delete[] reinterpret_cast<char *>(hostAddr);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
      delete[] reinterpret_cast<char *>(buf_ptrs[ID]);
    });

    break;
  } // END CASE OFFLOAD_EMISSARY_DM

  default: {
    return ::rpc::RPC_UNHANDLED_OPCODE;
    break;
  }
  }
  return ::rpc::RPC_SUCCESS;
} // end handle_emissary_impl

} // namespace internal

// Handles any opcode generated from emissary client code.
inline RPCStatus handleEmissaryOpcodes(Server::Port &port, uint32_t num_lanes) {
  switch (num_lanes) {
  case 1:
    return internal::handle_emissary_impl<1>(port);
  case 32:
    return internal::handle_emissary_impl<32>(port);
  case 64:
    return internal::handle_emissary_impl<64>(port);
  default:
    return RPC_ERROR;
  }
}

} // namespace rpc

#endif // LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
