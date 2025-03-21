// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fblocks -include %s -verify %s

// RUN: %clang -x c++ -fsyntax-only -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++11 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++20 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: [-Wunsafe-buffer-usage]

#ifndef INCLUDED
#define INCLUDED
#pragma clang system_header

// Xfail buffer warns until MIOPEN GTEST compiles ok
// XFAIL: *

// no spanification warnings for system headers
void foo(...);  // let arguments of `foo` to hold testing expressions
void testAsSystemHeader(char *p) {
  ++p;

  auto ap1 = p;
  auto ap2 = &p;

  foo(p[1],
      ap1[1],
      ap2[2][3]);
}

#else

void testIncrement(char *p) { // expected-warning{{'p' is an unsafe pointer used for buffer access}}
  ++p; // expected-note{{used in pointer arithmetic here}}
  p++; // expected-note{{used in pointer arithmetic here}}
  --p; // expected-note{{used in pointer arithmetic here}}
  p--; // expected-note{{used in pointer arithmetic here}}
}

void * voidPtrCall(void);
char * charPtrCall(void);

void testArraySubscripts(int idx, int *p, int **pp) {
// expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
// expected-warning@-2{{'pp' is an unsafe pointer used for buffer access}}
  foo(p[1],             // expected-note{{used in buffer access here}}
      pp[1][1],         // expected-note{{used in buffer access here}}
                        // expected-warning@-1{{unsafe buffer access}}
      1[1[pp]],         // expected-note{{used in buffer access here}}
                        // expected-warning@-1{{unsafe buffer access}}
      1[pp][1]          // expected-note{{used in buffer access here}}
                        // expected-warning@-1{{unsafe buffer access}}
      );

  if (p[3]) {           // expected-note{{used in buffer access here}}
    void * q = p;

    foo(((int*)q)[10]); // expected-warning{{unsafe buffer access}}
  }

  foo(((int*)voidPtrCall())[3], // expected-warning{{unsafe buffer access}}
      3[(int*)voidPtrCall()],   // expected-warning{{unsafe buffer access}}
      charPtrCall()[3],         // expected-warning{{unsafe buffer access}}
      3[charPtrCall()]          // expected-warning{{unsafe buffer access}}
      );

    int a[10];          // expected-warning{{'a' is an unsafe buffer that does not perform bounds checks}}
                        // expected-note@-1{{change type of 'a' to 'std::array' to label it for hardening}}
    int b[10][10];      // expected-warning{{'b' is an unsafe buffer that does not perform bounds checks}}

  foo(a[idx], idx[a],   // expected-note2{{used in buffer access here}}
      b[idx][idx + 1],  // expected-warning{{unsafe buffer access}}
                        // expected-note@-1{{used in buffer access here}}
      (idx + 1)[b][idx],// expected-warning{{unsafe buffer access}}
                        // expected-note@-1{{used in buffer access here}}
      (idx + 1)[idx[b]]);
                        // expected-warning@-1{{unsafe buffer access}}
                        // expected-note@-2{{used in buffer access here}}

  // Not to warn when index is zero
  foo(p[0], pp[0][0], 0[0[pp]], 0[pp][0],
      ((int*)voidPtrCall())[0],
      0[(int*)voidPtrCall()],
      charPtrCall()[0],
      0[charPtrCall()]
      );
}

void testArraySubscriptsWithAuto() {
  int a[10];
  // We do not fix a declaration if the type is `auto`. Because the actual type may change later.
  auto ap1 = a;   // expected-warning{{'ap1' is an unsafe pointer used for buffer access}}
  foo(ap1[1]);    // expected-note{{used in buffer access here}}

  // In case the type is `auto *`, we know it must be a pointer. We can fix it.
  auto * ap2 = a; // expected-warning{{'ap2' is an unsafe pointer used for buffer access}} \
                     expected-note{{change type of 'ap2' to 'std::span' to preserve bounds information}}
  foo(ap2[1]);    // expected-note{{used in buffer access here}}
}

void testUnevaluatedContext(int * p) {// no-warning
  foo(sizeof(p[1]),             // no-warning
      sizeof(decltype(p[1])));  // no-warning
}

void testQualifiedParameters(const int * p, const int * const q, const int a[10], const int b[10][10]) {
  // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
  // expected-warning@-2{{'q' is an unsafe pointer used for buffer access}}
  // expected-warning@-3{{'a' is an unsafe pointer used for buffer access}}
  // expected-warning@-4{{'b' is an unsafe pointer used for buffer access}}

  foo(p[1], 1[p], p[-1],    // expected-note3{{used in buffer access here}}
      q[1], 1[q], q[-1],    // expected-note3{{used in buffer access here}}
      a[1],                 // expected-note{{used in buffer access here}}     `a` is of pointer type
      b[1][2]               // expected-note{{used in buffer access here}}     `b[1]` is of array type
      );
}

struct T {
  int a[10];
  int * b;
  struct {
    int a[10];
    int * b;
  } c;
};

typedef struct T T_t;

T_t funRetT();
T_t * funRetTStar();

void testStructMembers(struct T * sp, struct T s, T_t * sp2, T_t s2) {
  foo(sp->a[1],
      sp->b[1],     // expected-warning{{unsafe buffer access}}
      sp->c.a[1],
      sp->c.b[1],   // expected-warning{{unsafe buffer access}}
      s.a[1],
      s.b[1],       // expected-warning{{unsafe buffer access}}
      s.c.a[1],
      s.c.b[1],     // expected-warning{{unsafe buffer access}}
      sp2->a[1],
      sp2->b[1],    // expected-warning{{unsafe buffer access}}
      sp2->c.a[1],
      sp2->c.b[1],  // expected-warning{{unsafe buffer access}}
      s2.a[1],
      s2.b[1],      // expected-warning{{unsafe buffer access}}
      s2.c.a[1],
      s2.c.b[1],           // expected-warning{{unsafe buffer access}}
      funRetT().a[1],
      funRetT().b[1],      // expected-warning{{unsafe buffer access}}
      funRetTStar()->a[1],
      funRetTStar()->b[1]  // expected-warning{{unsafe buffer access}}
      );
}

union Foo {
   bool b;
   int arr[10];
};

int testUnionMembers(Foo f) {
  int a = f.arr[0];
  a = f.arr[5];
  a = f.arr[10]; // expected-warning{{unsafe buffer access}}
  return a;
}

int garray[10];     // expected-warning{{'garray' is an unsafe buffer that does not perform bounds checks}}
int * gp = garray;  // expected-warning{{'gp' is an unsafe pointer used for buffer access}}
int gvar = gp[1];   // FIXME: file scope unsafe buffer access is not warned

void testLambdaCaptureAndGlobal(int * p) {
  // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
  int a[10];              // expected-warning{{'a' is an unsafe buffer that does not perform bounds checks}}

  auto Lam = [p, a](int idx) {
    return p[1]           // expected-note{{used in buffer access here}}
      + a[idx] + garray[idx]// expected-note2{{used in buffer access here}}
      + gp[1];            // expected-note{{used in buffer access here}}

  };
}

auto file_scope_lambda = [](int *ptr) {
  // expected-warning@-1{{'ptr' is an unsafe pointer used for buffer access}}
  
  ptr[5] = 10;  // expected-note{{used in buffer access here}}
};

void testLambdaCapture() {
  int a[10];              // expected-warning{{'a' is an unsafe buffer that does not perform bounds checks}}
  int b[10];              // expected-warning{{'b' is an unsafe buffer that does not perform bounds checks}}
                          // expected-note@-1{{change type of 'b' to 'std::array' to label it for hardening}}
  int c[10];

  auto Lam1 = [a](unsigned idx) {
    return a[idx];           // expected-note{{used in buffer access here}}
  };

  auto Lam2 = [x = b[c[5]]]() { // expected-note{{used in buffer access here}}
    return x;
  };

  auto Lam = [x = c](unsigned idx) { // expected-warning{{'x' is an unsafe pointer used for buffer access}}
    return x[idx]; // expected-note{{used in buffer access here}}
  };
}

void testLambdaImplicitCapture(long idx) {
  int a[10];              // expected-warning{{'a' is an unsafe buffer that does not perform bounds checks}}
                          // expected-note@-1{{change type of 'a' to 'std::array' to label it for hardening}}
  int b[10];              // expected-warning{{'b' is an unsafe buffer that does not perform bounds checks}}
                          // expected-note@-1{{change type of 'b' to 'std::array' to label it for hardening}}
  
  auto Lam1 = [=]() {
    return a[idx];           // expected-note{{used in buffer access here}}
  };
  
  auto Lam2 = [&]() {
    return b[idx];           // expected-note{{used in buffer access here}}
  };
}

typedef T_t * T_ptr_t;

void testTypedefs(T_ptr_t p) {
  // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
  foo(p[1],       // expected-note{{used in buffer access here}}
      p[1].a[1],  // expected-note{{used in buffer access here}}
      p[1].b[1]   // expected-note{{used in buffer access here}}
                  // expected-warning@-1{{unsafe buffer access}}
      );
}

template<typename T, int N> T f(T t, T * pt, T a[N], T (&b)[N]) {
  // expected-warning@-1{{'t' is an unsafe pointer used for buffer access}}
  // expected-warning@-2{{'pt' is an unsafe pointer used for buffer access}}
  // expected-warning@-3{{'a' is an unsafe pointer used for buffer access}}
  foo(pt[1],    // expected-note{{used in buffer access here}}
      a[1],     // expected-note{{used in buffer access here}}
      b[1]);
  return &t[1]; // expected-note{{used in buffer access here}}
}

// Testing pointer arithmetic for pointer-to-int, qualified multi-level
// pointer, pointer to a template type, and auto type
T_ptr_t getPtr();

template<typename T>
void testPointerArithmetic(int * p, const int **q, T * x) {
// expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
// expected-warning@-2{{'x' is an unsafe pointer used for buffer access}}
  int a[10];
  auto y = &a[0]; // expected-warning{{'y' is an unsafe pointer used for buffer access}}

  foo(p + 1, 1 + p, p - 1,      // expected-note3{{used in pointer arithmetic here}}
      *q + 1, 1 + *q, *q - 1,   // expected-warning3{{unsafe pointer arithmetic}}
      x + 1, 1 + x, x - 1,      // expected-note3{{used in pointer arithmetic here}}
      y + 1, 1 + y, y - 1,      // expected-note3{{used in pointer arithmetic here}}
      getPtr() + 1, 1 + getPtr(), getPtr() - 1 // expected-warning3{{unsafe pointer arithmetic}}
      );

  p += 1;  p -= 1;  // expected-note2{{used in pointer arithmetic here}}
  *q += 1; *q -= 1; // expected-warning2{{unsafe pointer arithmetic}}
  y += 1; y -= 1;   // expected-note2{{used in pointer arithmetic here}}
  x += 1; x -= 1;   // expected-note2{{used in pointer arithmetic here}}
}

void testTemplate(int * p) {
  int *a[10];
  foo(f(p, &p, a, a)[1]); // expected-warning{{unsafe buffer access}}
                          // expected-note@-1{{in instantiation of function template specialization 'f<int *, 10>' requested here}}

  const int **q = const_cast<const int **>(&p);

  testPointerArithmetic(p, q, p); //expected-note{{in instantiation of}}
}

void testPointerToMember() {
  struct S_t {
    int x;
    int * y;
  } S;

  int S_t::* p = &S_t::x;
  int * S_t::* q = &S_t::y;

  foo(S.*p,
      (S.*q)[1]);  // expected-warning{{unsafe buffer access}}
}

// test that nested callable definitions are scanned only once
void testNestedCallableDefinition(int * p) {
  class A {
    void inner(int * p) {
      // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
      p++; // expected-note{{used in pointer arithmetic here}}
    }

    static void innerStatic(int * p) {
      // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
      p++; // expected-note{{used in pointer arithmetic here}}
    }

    void innerInner(int * p) {
      auto Lam = [p]() {
        int * q = p;    // expected-warning{{'q' is an unsafe pointer used for buffer access}}
        q++;            // expected-note{{used in pointer arithmetic here}}
        return *q;
      };
    }
  };

  auto Lam = [p]() {
    int * q = p;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}
    q++;          // expected-note{{used in pointer arithmetic here}}
    return *q;
  };

  auto LamLam = [p]() {
    auto Lam = [p]() {
      int * q = p;  // expected-warning{{'q' is an unsafe pointer used for buffer access}}
      q++;          // expected-note{{used in pointer arithmetic here}}
      return *q;
    };
  };

  void (^Blk)(int*) = ^(int *p) {
    // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
    p++;   // expected-note{{used in pointer arithmetic here}}
  };

  void (^BlkBlk)(int*) = ^(int *p) {
    void (^Blk)(int*) = ^(int *p) {
      // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
      p++;   // expected-note{{used in pointer arithmetic here}}
    };
    Blk(p);
  };

  // lambda and block as call arguments...
  foo( [p]() { int * q = p; // expected-warning{{'q' is an unsafe pointer used for buffer access}}
              q++;          // expected-note{{used in pointer arithmetic here}}
              return *q;
       },
       ^(int *p) {  // expected-warning{{'p' is an unsafe pointer used for buffer access}}
        p++;        // expected-note{{used in pointer arithmetic here}}
       }
     );
}

int testVariableDecls(int * p) {
  // expected-warning@-1{{'p' is an unsafe pointer used for buffer access}}
  int * q = p++;      // expected-note{{used in pointer arithmetic here}}
  int a[p[1]];        // expected-note{{used in buffer access here}}
  int b = p[1];       // expected-note{{used in buffer access here}}
  return p[1];        // expected-note{{used in buffer access here}}
}

template<typename T> void fArr(T t[], long long idx) {
  // expected-warning@-1{{'t' is an unsafe pointer used for buffer access}}
  foo(t[1]);    // expected-note{{used in buffer access here}}
  T ar[8];      // expected-warning{{'ar' is an unsafe buffer that does not perform bounds checks}}
                // expected-note@-1{{change type of 'ar' to 'std::array' to label it for hardening}}
  foo(ar[idx]);   // expected-note{{used in buffer access here}}
}

<<<<<<< HEAD
template void fArr<int>(int t[]); // expected-note {{in instantiation of}}
=======
template void fArr<int>(int t[], long long); // FIXME: expected note {{in instantiation of}}
>>>>>>> 594d57e07a92e3a2cefb262114db2608989f874d

int testReturn(int t[]) {// expected-note{{change type of 't' to 'std::span' to preserve bounds information}}
  // expected-warning@-1{{'t' is an unsafe pointer used for buffer access}}
  return t[1]; // expected-note{{used in buffer access here}}
}

int testArrayAccesses(int n, int idx) {
    // auto deduced array type
    int cArr[2][3] = {{1, 2, 3}, {4, 5, 6}};
    // expected-warning@-1{{'cArr' is an unsafe buffer that does not perform bounds checks}}
    int d = cArr[0][0];
    foo(cArr[0][0]);
    foo(cArr[idx][idx + 1]);        // expected-note{{used in buffer access here}}
                                    // expected-warning@-1{{unsafe buffer access}}
    auto cPtr = cArr[idx][idx * 2]; // expected-note{{used in buffer access here}}
                                    // expected-warning@-1{{unsafe buffer access}}
    foo(cPtr);

    // Typdefs
    typedef int A[3];
    const A tArr = {4, 5, 6};
    foo(tArr[0], tArr[1]);
    return cArr[0][1];
}

void testArrayPtrArithmetic(int x[]) { // expected-warning{{'x' is an unsafe pointer used for buffer access}}
  foo (x + 3); // expected-note{{used in pointer arithmetic here}}

  int y[3] = {0, 1, 2}; // expected-warning{{'y' is an unsafe buffer that does not perform bounds checks}}
  foo(y + 4); // expected-note{{used in pointer arithmetic here}}
}

void testMultiLineDeclStmt(int * p) {
  int

  *

  ap1 = p;      // expected-warning{{'ap1' is an unsafe pointer used for buffer access}} \
         	   expected-note{{change type of 'ap1' to 'std::span' to preserve bounds information}}

  foo(ap1[1]);  // expected-note{{used in buffer access here}}
}

#endif
