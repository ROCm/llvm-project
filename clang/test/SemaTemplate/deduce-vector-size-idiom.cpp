// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++17 -fsyntax-only -verify %s

template <class A, class B> struct is_same {
  static constexpr bool value = false;
};
template <class A> struct is_same<A, A> {
  static constexpr bool value = true;
};

template <class T> T make();
template <int N> struct tag {};

template <class T, int N>
using gen = T __attribute__((__vector_size__(N * sizeof(T))));
template <class T, int N>
using ext = T __attribute__((ext_vector_type(N)));

namespace both_deduced {
template <class V> struct traits {
  static constexpr int size = 0;
  using scalar = void;
};
template <class T, int N>
struct traits<T __attribute__((__vector_size__(N * sizeof(T))))> {
  static constexpr int size = N;
  using scalar = T;
};

static_assert(traits<gen<float, 4>>::size == 4);
static_assert(is_same<traits<gen<float, 4>>::scalar, float>::value);
static_assert(traits<gen<double, 2>>::size == 2);

static_assert(traits<gen<char, 16>>::size == 16);
static_assert(is_same<traits<gen<char, 16>>::scalar, char>::value);

static_assert(traits<float>::size == 0);
static_assert(traits<ext<float, 4>>::size == 0);
}

namespace spelling {
template <class V> struct reversed {
  static constexpr int size = 0;
};
template <class T, int N>
struct reversed<T __attribute__((__vector_size__(sizeof(T) * N)))> {
  static constexpr int size = N;
};
static_assert(reversed<gen<int, 2>>::size == 2);
static_assert(reversed<gen<double, 4>>::size == 4);

template <class V> struct parenthesized {
  static constexpr int size = 0;
};
template <class T, int N>
struct parenthesized<T __attribute__((__vector_size__((N * sizeof(T)))))> {
  static constexpr int size = N;
};
static_assert(parenthesized<gen<int, 4>>::size == 4);
}

namespace fixed_element {
template <class V> struct traits {
  static constexpr int size = 0;
};
template <int N>
struct traits<int __attribute__((__vector_size__(N * sizeof(int))))> {
  static constexpr int size = N;
};
static_assert(traits<gen<int, 4>>::size == 4);
static_assert(traits<gen<int, 16>>::size == 16);
static_assert(traits<gen<float, 4>>::size == 0);
}

namespace unsigned_parameter {
template <class V> struct traits {
  static constexpr unsigned size = 0;
};
template <class T, unsigned N>
struct traits<T __attribute__((__vector_size__(N * sizeof(T))))> {
  static constexpr unsigned size = N;
};
static_assert(traits<gen<int, 4>>::size == 4u);
}

namespace via_alias {
template <class T, int Rank>
using native_vector = T __attribute__((__vector_size__(Rank * sizeof(T))));

template <class V> struct traits {
  static constexpr int size = 0;
  using scalar = void;
};
template <class T, int Rank>
struct traits<native_vector<T, Rank>> {
  static constexpr int size = Rank;
  using scalar = T;
};

static_assert(traits<native_vector<float, 4>>::size == 4);
static_assert(traits<gen<double, 8>>::size == 8);
static_assert(is_same<traits<gen<double, 8>>::scalar, double>::value);
static_assert(traits<float>::size == 0);
}

namespace variable_template {
template <class V> constexpr int size = 0;
template <class T, int N>
constexpr int size<T __attribute__((__vector_size__(N * sizeof(T))))> = N;

static_assert(size<gen<float, 4>> == 4);
static_assert(size<gen<char, 16>> == 16);
static_assert(size<float> == 0);
}

namespace coexistence {
struct ExtKind {};
struct GenKind {};

template <class V> struct notation {
  static constexpr int size = 0;
  using kind = void;
};
template <class T, int N>
struct notation<T __attribute__((ext_vector_type(N)))> {
  static constexpr int size = N;
  using kind = ExtKind;
};
template <class T, int N>
struct notation<T __attribute__((__vector_size__(N * sizeof(T))))> {
  static constexpr int size = N;
  using kind = GenKind;
};

static_assert(notation<ext<float, 4>>::size == 4);
static_assert(is_same<notation<ext<float, 4>>::kind, ExtKind>::value);
static_assert(notation<gen<float, 4>>::size == 4);
static_assert(is_same<notation<gen<float, 4>>::kind, GenKind>::value);
static_assert(notation<float>::size == 0);
}

namespace function_templates {
template <class T, int N> tag<N> by_value(gen<T, N>);
static_assert(
    is_same<decltype(by_value(make<gen<float, 4>>())), tag<4>>::value);
static_assert(is_same<decltype(by_value(make<gen<char, 8>>())), tag<8>>::value);

template <int N> tag<N> fixed_element(gen<float, N>);
static_assert(
    is_same<decltype(fixed_element(make<gen<float, 8>>())), tag<8>>::value);

template <class T, int N>
tag<N> reversed(T __attribute__((__vector_size__(sizeof(T) * N))));
static_assert(is_same<decltype(reversed(make<gen<int, 4>>())), tag<4>>::value);

template <class T, int N> tag<N> by_ref(const gen<T, N> &);
static_assert(is_same<decltype(by_ref(make<gen<float, 4>>())), tag<4>>::value);

template <class T, int N> tag<N> by_ptr(gen<T, N> *);
static_assert(
    is_same<decltype(by_ptr(make<gen<float, 4> *>())), tag<4>>::value);

template <class T, int N> tag<N> ext_vec(ext<T, N>);
static_assert(is_same<decltype(ext_vec(make<ext<float, 4>>())), tag<4>>::value);
}

namespace not_the_idiom {
template <class V> struct traits {
  static constexpr int size = 0;
};

// expected-note@+2{{non-deducible template parameter 'N'}}
// expected-error@+2{{contains a template parameter that cannot be deduced}}
template <class T, int N>
struct traits<T __attribute__((__vector_size__(N * 4)))> {
  static constexpr int size = N;
};

// expected-note@+2{{non-deducible template parameter 'N'}}
// expected-error@+2{{contains a template parameter that cannot be deduced}}
template <class T, int N>
struct traits<T __attribute__((__vector_size__(N * sizeof(int))))> {
  static constexpr int size = N;
};

// expected-note@+2{{non-deducible template parameter 'N'}}
// expected-error@+2{{contains a template parameter that cannot be deduced}}
template <class T, int N>
struct traits<T __attribute__((__vector_size__(N + sizeof(T))))> {
  static constexpr int size = N;
};

// expected-note@+2{{non-deducible template parameter 'N'}}
// expected-error@+2{{contains a template parameter that cannot be deduced}}
template <class T, int N>
struct traits<T __attribute__((__vector_size__(2 * N * sizeof(T))))> {
  static constexpr int size = N;
};

static_assert(traits<gen<float, 4>>::size == 0);
static_assert(traits<gen<int, 4>>::size == 0);
static_assert(traits<gen<float, 8>>::size == 0);
}

namespace bare_size {
template <class V> struct traits {
  static constexpr int size = 0;
};
template <class T, int Bytes>
struct traits<T __attribute__((__vector_size__(Bytes)))> {
  static constexpr int size = Bytes;
};
static_assert(traits<gen<char, 16>>::size == 16);
static_assert(traits<gen<float, 4>>::size == 0);
}

namespace no_deducible_size {
template <class V> struct traits {
  static constexpr bool matched = false;
};
template <class T>
struct traits<T __attribute__((__vector_size__(sizeof(T) * sizeof(T))))> {
  static constexpr bool matched = true;
};
static_assert(traits<gen<float, 4>>::matched);
static_assert(!traits<gen<double, 2>>::matched);
}
