//===----- ioffload/plugins-nexgen/common/include/Emissary.cpp ---- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPC.h"

#include "Shared/Debug.h"
#include "Shared/RPCOpcodes.h"

#include "PluginInterface.h"

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

#include "../../../DeviceRTL/include/EmissaryIds.h"
#include "Emissary.h"

extern "C" emis_return_t Emissary(char *data) {
  emisArgBuf_t ab;
  emisExtractArgBuf(data, &ab);
  emis_return_t result = 0;
  emis_argptr_t *args[MAXVARGS]; // FIXME use malloc here

  switch (ab.emisid) {
  case EMIS_ID_INVALID: {
    fprintf(stderr, "emisExecute got invalid EMIS_ID\n");
    result = 0;
    break;
  }
  case EMIS_ID_FORTRT: {
    result = EmissaryFortrt(data, &ab);
    break;
  }
  case EMIS_ID_PRINT: {
    result = EmissaryPrint(data, &ab);
    break;
  }
  case EMIS_ID_MPI: {
    if (EmissaryBuildVargs(ab.NumArgs, ab.keyptr, ab.argptr, ab.strptr,
                           &ab.data_not_used, &args[0]) != _RC_SUCCESS)
      return (emis_return_t)0;
    result = EmissaryMPI(data, &ab, args);
    break;
  }
  case EMIS_ID_HDF5: {
    if (EmissaryBuildVargs(ab.NumArgs, ab.keyptr, ab.argptr, ab.strptr,
                           &ab.data_not_used, &args[0]) != _RC_SUCCESS)
      return (emis_return_t)0;
    result = EmissaryHDF5(data, &ab, args);
    break;
  }
  case EMIS_ID_RESERVE: {
    if (EmissaryBuildVargs(ab.NumArgs, ab.keyptr, ab.argptr, ab.strptr,
                           &ab.data_not_used, &args[0]) != _RC_SUCCESS)
      return (emis_return_t)0;
    result = EmissaryReserve(data, &ab, args);
    break;
  }
  default:
    fprintf(stderr, "EMIS_ID:%d fnid:%d not supported\n", ab.emisid,
            ab.emisfnid);
  }
  return result;
}

// emisExtractArgBuf reverses protocol that codegen in EmitEmissaryExec makes.
extern "C" void emisExtractArgBuf(char *data, emisArgBuf_t *ab) {

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

  // Extract the two emissary identifiers from 1st 64bit arg
  uint64_t emisIds = *(uint64_t *)ab->argptr;
  ab->emisid = (offload_emis_id_t)((uint)(emisIds >> 32));
  ab->emisfnid = (uint32_t)((uint)((emisIds << 32) >> 32));

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
extern "C" uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}

/// Get uint64_t value from a char ptr
extern "C" uint64_t getuint64(char *val) { return *(uint64_t *)val; }

/// Get a function pointer from a char ptr
extern "C" void *getfnptr(char *val) {
  uint64_t ival = *(uint64_t *)val;
  return (void *)ival;
}

// build argument array
extern "C" uint32_t EmissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr,
                                       char *strptr,
                                       unsigned long long *data_not_used,
                                       emis_argptr_t *a[MAXVARGS]) {
  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  uint argcount = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);

      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);

      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;
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
          return _RC_DATA_USED_ERROR;

        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case TypedPointerTyID:   ///< Typed pointer used by some GPU targets
    case TargetExtTyID:      ///< Target extension type
    case VoidTyID:
      return _RC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _RC_INVALID_ID_ERROR;
    }

    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return _RC_SUCCESS;
}

// Host defines for f90print functions needed just for linking
// and fallback when used in a target region
extern "C" void f90print_(char *s) { printf("%s\n", s); }
extern "C" void f90printi_(char *s, int *i) { printf("%s %d\n", s, *i); }
extern "C" void f90printl_(char *s, long *i) { printf("%s %ld\n", s, *i); }
extern "C" void f90printf_(char *s, float *f) { printf("%s %f\n", s, *f); }
extern "C" void f90printd_(char *s, double *d) { printf("%s %g\n", s, *d); }

extern "C" void *rpc_allocate(uint64_t sz) {
  printf("HOST rpc_allocate\n");
  return nullptr;
}
extern "C" void rpc_free(void *ptr) {
  printf("HOST rpc_free\n");
  return;
}
