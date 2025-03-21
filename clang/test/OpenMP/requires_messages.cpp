// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100  %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=99 -DOMP99 -verify=expected,rev -ferror-limit 100  %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -DOMP60 -verify=self -ferror-limit 100  %s -Wuninitialized

int a;
#pragma omp requires unified_address allocate(a) // rev-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note {{unified_address clause previously used here}} expected-note{{unified_address clause previously used here}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp requires'}}

#pragma omp requires unified_shared_memory // rev-note {{unified_shared_memory clause previously used here}} expected-note{{unified_shared_memory clause previously used here}}

#pragma omp requires unified_shared_memory, unified_shared_memory // expected-error {{only one unified_shared_memory clause can appear on a requires directive in a single translation unit}} expected-error {{directive '#pragma omp requires' cannot contain more than one 'unified_shared_memory' clause}}

#pragma omp requires unified_address // expected-error {{only one unified_address clause can appear on a requires directive in a single translation unit}}

#pragma omp requires unified_address, unified_address // expected-error {{only one unified_address clause can appear on a requires directive in a single translation unit}} expected-error {{directive '#pragma omp requires' cannot contain more than one 'unified_address' clause}}

#ifdef OMP99
#pragma omp requires reverse_offload // rev-note {{reverse_offload clause previously used here}} rev-note {{reverse_offload clause previously used here}}

#pragma omp requires reverse_offload, reverse_offload // rev-error {{only one reverse_offload clause can appear on a requires directive in a single translation unit}} rev-error {{directive '#pragma omp requires' cannot contain more than one 'reverse_offload' clause}}
#endif

#pragma omp requires dynamic_allocators // rev-note {{dynamic_allocators clause previously used here}} expected-note {{dynamic_allocators clause previously used here}}

#pragma omp requires dynamic_allocators, dynamic_allocators // expected-error {{only one dynamic_allocators clause can appear on a requires directive in a single translation unit}} expected-error {{directive '#pragma omp requires' cannot contain more than one 'dynamic_allocators' clause}}

#ifdef OMP60
#pragma omp requires self_maps // self-note {{self_maps clause previously used here}}

#pragma omp requires self_maps, self_maps // self-error {{only one self_maps clause can appear on a requires directive in a single translation unit}} self-error {{directive '#pragma omp requires' cannot contain more than one 'self_maps' clause}}
#endif

#pragma omp requires atomic_default_mem_order(seq_cst) // rev-note {{atomic_default_mem_order clause previously used here}} expected-note {{atomic_default_mem_order clause previously used here}} expected-note {{atomic_default_mem_order clause previously used here}} expected-note {{atomic_default_mem_order clause previously used here}} expected-note {{atomic_default_mem_order clause previously used here}}

#pragma omp requires atomic_default_mem_order(acq_rel) // expected-error {{only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}

#pragma omp requires atomic_default_mem_order(relaxed) // expected-error {{only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}

#pragma omp requires atomic_default_mem_order // expected-error {{expected '(' after 'atomic_default_mem_order'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires atomic_default_mem_order( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'seq_cst', 'acq_rel' or 'relaxed' in OpenMP clause 'atomic_default_mem_order'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires atomic_default_mem_order(seq_cst // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}

#pragma omp requires atomic_default_mem_order(invalid_modifier) // expected-error {{expected 'seq_cst', 'acq_rel' or 'relaxed' in OpenMP clause 'atomic_default_mem_order'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires atomic_default_mem_order(shared) // expected-error {{expected 'seq_cst', 'acq_rel' or 'relaxed' in OpenMP clause 'atomic_default_mem_order'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires atomic_default_mem_order(acq_rel), atomic_default_mem_order(relaxed) // expected-error {{directive '#pragma omp requires' cannot contain more than one 'atomic_default_mem_order' claus}} expected-error {{only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}

#pragma omp requires // expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp requires'}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#pragma omp requires unified_address, invalid_clause // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{only one unified_address clause can appear on a requires directive in a single translation unit}}

#pragma omp requires invalid_clause unified_address // expected-warning {{extra tokens at the end of '#pragma omp requires' are ignored}} expected-error {{expected at least one clause on '#pragma omp requires' directive}}

#ifdef OMP99
#pragma omp requires unified_shared_memory, unified_address, reverse_offload, dynamic_allocators, atomic_default_mem_order(seq_cst) // rev-error {{only one unified_shared_memory clause can appear on a requires directive in a single translation unit}} rev-error{{only one unified_address clause can appear on a requires directive in a single translation unit}} rev-error{{only one reverse_offload clause can appear on a requires directive in a single translation unit}} rev-error{{only one dynamic_allocators clause can appear on a requires directive in a single translation unit}} rev-error {{only one atomic_default_mem_order clause can appear on a requires directive in a single translation unit}}
#endif

namespace A {
  #pragma omp requires unified_address // expected-error {{only one unified_address clause can appear on a requires directive in a single translation unit}}
  namespace B {
    #pragma omp requires unified_address // expected-error {{only one unified_address clause can appear on a requires directive in a single translation unit}}
  }
}

template <typename T> T foo() {
  #pragma omp requires unified_address // expected-error {{unexpected OpenMP directive '#pragma omp requires'}}
}

class C {
  #pragma omp requires unified_address // expected-error {{'#pragma omp requires' directive must appear only in file scope}}
};

int main() {
  #pragma omp requires unified_address // expected-error {{unexpected OpenMP directive '#pragma omp requires'}}
}
