//===- symbols_edge_test.c ------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Covers the symbol lookup and iteration paths that the happy-path tests miss:
/// buffers that are not object files, truncated objects, absent symbol names,
/// and the difference between how a relocatable and an executable are searched
/// (static symbol table versus dynamic symbol table).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static amd_comgr_data_t makeData(amd_comgr_data_kind_t Kind, const char *Bytes,
                                 size_t Size) {
  amd_comgr_data_t Data;
  checkError(amd_comgr_create_data(Kind, &Data), "amd_comgr_create_data");
  checkError(amd_comgr_set_data(Data, Size, Bytes), "amd_comgr_set_data");
  return Data;
}

static amd_comgr_status_t countSymbol(amd_comgr_symbol_t Symbol,
                                      void *UserData) {
  size_t NameLength = 0;
  amd_comgr_status_t Status = amd_comgr_symbol_get_info(
      Symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &NameLength);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return Status;
  *(size_t *)UserData += 1;
  return AMD_COMGR_STATUS_SUCCESS;
}

// A buffer that is not an object file at all must be rejected by both entry
// points rather than parsed.
static void testNotAnObject(void) {
  const char *Garbage = "this is definitely not an ELF file";
  amd_comgr_data_t Data;
  amd_comgr_symbol_t Symbol;
  size_t Count = 0;

  Data = makeData(AMD_COMGR_DATA_KIND_EXECUTABLE, Garbage, strlen(Garbage));

  checkStatus(amd_comgr_symbol_lookup(Data, "foo", &Symbol),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_symbol_lookup on a non-object buffer");
  checkStatus(amd_comgr_iterate_symbols(Data, countSymbol, &Count),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_iterate_symbols on a non-object buffer");
  if (Count != 0)
    fail("iterate_symbols invoked the callback for a non-object buffer");

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
}

// An ELF header followed by nothing: the magic matches but the section and
// symbol tables are unreachable.
static void testTruncatedObject(const char *Path) {
  char *Buf;
  long Size;
  amd_comgr_data_t Data;
  amd_comgr_symbol_t Symbol;

  Size = setBuf(Path, &Buf);
  if (Size < 64)
    fail("%s is too small to truncate meaningfully", Path);

  Data = makeData(AMD_COMGR_DATA_KIND_EXECUTABLE, Buf, 48);
  checkStatus(amd_comgr_symbol_lookup(Data, "foo", &Symbol),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_symbol_lookup on a truncated object");
  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");

  free(Buf);
}

// A name that is not in the symbol table is an error, and asking twice must
// behave the same both times.
static void testMissingSymbol(const char *Path) {
  char *Buf;
  long Size;
  amd_comgr_data_t Data;
  amd_comgr_symbol_t Symbol;

  Size = setBuf(Path, &Buf);
  Data = makeData(AMD_COMGR_DATA_KIND_EXECUTABLE, Buf, Size);

  checkStatus(amd_comgr_symbol_lookup(Data, "no_such_symbol_at_all", &Symbol),
              AMD_COMGR_STATUS_ERROR, "amd_comgr_symbol_lookup missing symbol");
  checkStatus(amd_comgr_symbol_lookup(Data, "no_such_symbol_at_all", &Symbol),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_symbol_lookup missing symbol, repeated");
  // The empty name is not a symbol a caller can look up either.
  checkStatus(amd_comgr_symbol_lookup(Data, "", &Symbol),
              AMD_COMGR_STATUS_ERROR, "amd_comgr_symbol_lookup empty name");

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  free(Buf);
}

// A relocatable is searched through its static symbol table, an executable
// through its dynamic one. Both paths must enumerate and resolve.
static void testRelocatableAndExecutable(void) {
  struct {
    const char *Path;
    amd_comgr_data_kind_t Kind;
  } Cases[] = {
      {TEST_OBJ_DIR "/reloc1.o", AMD_COMGR_DATA_KIND_RELOCATABLE},
      {TEST_OBJ_DIR "/shared-v3.so", AMD_COMGR_DATA_KIND_EXECUTABLE},
  };

  for (size_t I = 0; I < sizeof(Cases) / sizeof(Cases[0]); ++I) {
    char *Buf;
    long Size;
    amd_comgr_data_t Data;
    size_t Count = 0;

    Size = setBuf(Cases[I].Path, &Buf);
    Data = makeData(Cases[I].Kind, Buf, Size);

    checkError(amd_comgr_iterate_symbols(Data, countSymbol, &Count),
               "amd_comgr_iterate_symbols");
    if (Count == 0)
      fail("%s produced no symbols", Cases[I].Path);

    // Iterating twice must produce the same count.
    size_t Again = 0;
    checkError(amd_comgr_iterate_symbols(Data, countSymbol, &Again),
               "amd_comgr_iterate_symbols repeated");
    if (Again != Count)
      fail("%s enumerated %zu symbols then %zu", Cases[I].Path, Count, Again);

    checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
    free(Buf);
  }
}

// The data kind selects which table is searched: EXECUTABLE consults the
// dynamic symbol table, RELOCATABLE the static one. This shared object is not
// stripped, so the kernel appears in both and each path can be checked to
// resolve it to the same address.
static void testKindSelectsSymbolTable(void) {
  const char *KernelName =
      "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
  const amd_comgr_data_kind_t Kinds[] = {AMD_COMGR_DATA_KIND_EXECUTABLE,
                                         AMD_COMGR_DATA_KIND_RELOCATABLE};
  char *Buf;
  long Size;
  uint64_t Values[2];

  Size = setBuf(TEST_OBJ_DIR "/shared-v3.so", &Buf);

  for (size_t I = 0; I < 2; ++I) {
    amd_comgr_data_t Data = makeData(Kinds[I], Buf, Size);
    amd_comgr_symbol_t Symbol;

    checkError(amd_comgr_symbol_lookup(Data, KernelName, &Symbol),
               "amd_comgr_symbol_lookup");
    checkError(amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_VALUE,
                                         &Values[I]),
               "AMD_COMGR_SYMBOL_INFO_VALUE");
    checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  }

  if (Values[0] != Values[1])
    fail("the dynamic and static symbol tables disagree on %s: %" PRIu64
         " vs %" PRIu64,
         KernelName, Values[0], Values[1]);

  free(Buf);
}

// Every documented symbol info field must be readable, and an out-of-range
// field must be rejected.
static void testSymbolInfoFields(void) {
  const char *KernelName =
      "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
  char *Buf;
  long Size;
  amd_comgr_data_t Data;
  amd_comgr_symbol_t Symbol;
  amd_comgr_symbol_type_t Type;
  uint64_t Value, SymbolSize;
  size_t NameLength;
  char *Name;
  bool Undefined;

  Size = setBuf(TEST_OBJ_DIR "/shared-v3.so", &Buf);
  Data = makeData(AMD_COMGR_DATA_KIND_EXECUTABLE, Buf, Size);

  checkError(amd_comgr_symbol_lookup(Data, KernelName, &Symbol),
             "amd_comgr_symbol_lookup");

  checkError(amd_comgr_symbol_get_info(Symbol,
                                       AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
                                       &NameLength),
             "AMD_COMGR_SYMBOL_INFO_NAME_LENGTH");
  if (NameLength != strlen(KernelName))
    fail("name length is %zu, expected %zu", NameLength, strlen(KernelName));

  Name = (char *)calloc(NameLength + 1, 1);
  if (!Name)
    fail("calloc");
  checkError(amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_NAME,
                                       Name),
             "AMD_COMGR_SYMBOL_INFO_NAME");
  if (strcmp(Name, KernelName))
    fail("symbol name is \"%s\", expected \"%s\"", Name, KernelName);
  free(Name);

  checkError(amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                       &Type),
             "AMD_COMGR_SYMBOL_INFO_TYPE");
  checkError(amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_SIZE,
                                       &SymbolSize),
             "AMD_COMGR_SYMBOL_INFO_SIZE");
  checkError(amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_VALUE,
                                       &Value),
             "AMD_COMGR_SYMBOL_INFO_VALUE");
  checkError(amd_comgr_symbol_get_info(
                 Symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED, &Undefined),
             "AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED");
  if (Undefined)
    fail("a defined kernel reported itself undefined");

  checkStatus(amd_comgr_symbol_get_info(
                  Symbol, (amd_comgr_symbol_info_t)0x1000, &Value),
              AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
              "amd_comgr_symbol_get_info with an out-of-range field");
  checkStatus(
      amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_VALUE, NULL),
      AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
      "amd_comgr_symbol_get_info with a null value");

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  free(Buf);
}

int main(int argc, char *argv[]) {
  testNotAnObject();
  testTruncatedObject(TEST_OBJ_DIR "/shared-v3.so");
  testMissingSymbol(TEST_OBJ_DIR "/shared-v3.so");
  testRelocatableAndExecutable();
  testKindSelectsSymbolTable();
  testSymbolInfoFields();
  printf("symbols_edge_test passed\n");
  return 0;
}
