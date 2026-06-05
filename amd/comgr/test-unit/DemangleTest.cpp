//===- DemangleTest.cpp - Unit tests for COMGR Demangle ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "gtest/gtest.h"

class DemangleTest
    : public ::testing::TestWithParam<std::tuple<const char *, const char *>> {};

TEST_P(DemangleTest, DemangleMatch) {
  auto [MangledName, ExpectedString] = GetParam();
  amd_comgr_data_t MangledData, DemangledData;

  ASSERT_COMGR(create_data(AMD_COMGR_DATA_KIND_BYTES, &MangledData));

  size_t Size = strlen(MangledName);
  ASSERT_COMGR(set_data(MangledData, Size, MangledName));
  ASSERT_COMGR(demangle_symbol_name(MangledData, &DemangledData));

  size_t DemangledSize = 0;
  ASSERT_COMGR(get_data(DemangledData, &DemangledSize, NULL));
  ASSERT_EQ(DemangledSize, strlen(ExpectedString));

  char *DemangledName = (char *)calloc(DemangledSize, sizeof(char));
  ASSERT_NE(DemangledName, nullptr);
  ASSERT_COMGR(get_data(DemangledData, &DemangledSize, DemangledName));
  ASSERT_TRUE(strncmp(DemangledName, ExpectedString, DemangledSize) == 0);

  free(DemangledName);
  ASSERT_COMGR(release_data(MangledData));
  ASSERT_COMGR(release_data(DemangledData));
}

INSTANTIATE_TEST_SUITE_P(DemangleTable, DemangleTest,
  ::testing::Values(
  // Tests from llvm/unittests/Demangle/DemangleTest.cpp
  std::make_tuple("_", "_"),
  std::make_tuple("_Z3fooi", "foo(int)"),
  std::make_tuple("__Z3fooi", "foo(int)"),
  std::make_tuple("___Z3fooi_block_invoke", "invocation function for block in foo(int)"),
  std::make_tuple("____Z3fooi_block_invoke", "invocation function for block in foo(int)"),
  std::make_tuple("?foo@@YAXH@Z", "void __cdecl foo(int)"),
  std::make_tuple("foo", "foo"),
  std::make_tuple("_RNvC3foo3bar", "foo::bar"),
  std::make_tuple("_Z3fooILi79EEbU7_ExtIntIXT_EEi", "bool foo<79>(int _ExtInt<79>)"),

  // Some additional test cases.
  std::make_tuple("_Znwm", "operator new(unsigned long)"),
  std::make_tuple("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::operator=(std::__cxx11::basic_string<char, "
       "std::char_traits<char>, std::allocator<char>> const&)"),
  std::make_tuple("_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_",
       "std::_Rb_tree_insert_and_rebalance(bool, std::_Rb_tree_node_base*, "
       "std::_Rb_tree_node_base*, std::_Rb_tree_node_base&)"),
  std::make_tuple("_ZSt17__throw_bad_allocv", "std::__throw_bad_alloc()"),
  std::make_tuple("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::~basic_string()"),
  std::make_tuple("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::~basic_string()"),
  std::make_tuple("_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base",
       "std::_Rb_tree_increment(std::_Rb_tree_node_base*)"),
  std::make_tuple("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::basic_string()"),
  std::make_tuple("_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__"
       "cxx1112basic_stringIS4_S5_T1_EE",
       "std::basic_ostream<char, std::char_traits<char>>& std::operator<<"
       "<char, std::char_traits<char>, std::allocator<char>"
       ">(std::basic_ostream<char, std::char_traits<char>>&, "
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>> const&)"),
  std::make_tuple("_ZdlPv", "operator delete(void*)"),
  std::make_tuple("_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc",
       "std::basic_ostream<char, std::char_traits<char>>& std::operator<<"
       "<std::char_traits<char>>(std::basic_ostream<char, "
       "std::char_traits<char>>&, char const*)"),
  std::make_tuple("_ZdlPvm", "operator delete(void*, unsigned long)"),
  std::make_tuple("_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base",
       "std::_Rb_tree_decrement(std::_Rb_tree_node_base*)"),
  std::make_tuple("_ZNSaIcED1Ev", "std::allocator<char>::~allocator()"),
  std::make_tuple("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1EPKcRKS3_",
       "std::__cxx11::basic_string<char, std::char_traits<char>, "
       "std::allocator<char>>::basic_string(char const*, std::allocator<char> "
       "const&)"),
  std::make_tuple("_ZNSt8ios_base4InitC1Ev", "std::ios_base::Init::Init()"),
  std::make_tuple("_ZNSolsEi", "std::ostream::operator<<(int)"),
  std::make_tuple("_ZNSaIcEC1Ev", "std::allocator<char>::allocator()")));
