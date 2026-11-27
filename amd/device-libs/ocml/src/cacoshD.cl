/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double4 MATH_PRIVATE(epcsqrtep)(double4 z);
extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);
extern CONSTATTR double MATH_PRIVATE(atan2ep)(double2 y, double2 x);

CONSTATTR double2
MATH_MANGLE(cacosh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double y = BUILTIN_ABS_F64(z.y);

    double2 l2, t;
    double2 tl = (double2)0.0;
    int e = 0;
    bool b = true;

    bool big = x >= 0x1.0p+54 || y >= 0x1.0p+54;

    // The near axis formulas below are accurate for y < 2^-27 * min(|x^2-1|, 1)
    double w = BUILTIN_MIN_F64(BUILTIN_ABS_F64(BUILTIN_FMA_F64(x, x, -1.0)), 1.0);

    if (!big && y > 0x1.0p-28 * w) {
        double4 z2p1 = (double4)(add(mul(add(y,x), sub(y,x)), 1.0), mul(y,x)*2.0);
        double4 rz2m1 = MATH_PRIVATE(epcsqrtep)(z2p1);
        rz2m1 = (double4)(csgn(rz2m1.hi, (double2)z.x), csgn(rz2m1.lo, (double2)z.y));
        double4 s = (double4)(add(rz2m1.lo, z.x), add(rz2m1.hi, z.y));
        l2 = add(sqr(s.lo), sqr(s.hi));
        t = (double2)(s.s1, z.y == 0.0 ? z.y : s.s3);
        tl = (double2)(s.s0, z.y == 0.0 ? 0.0 : s.s2);
    } else if (big) {
        e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(x,y));
        x = BUILTIN_FLDEXP_F64(x, -e);
        y = BUILTIN_FLDEXP_F64(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
        t = z;
    } else if (x < 1.0) {
        b = false;
        double2 r = root2(fsub(1.0, sqr(x)));
        // Scale a tiny y up so that the residual stays normal
        int kn = y < 0x1.0p-900 ? 512 : 0;
        double ns = BUILTIN_FLDEXP_F64(y, kn);
        l2 = con(BUILTIN_FLDEXP_F64(divh(ns, r), -kn), 0.0);
        t = (double2)(z.x, BUILTIN_COPYSIGN_F64(r.hi, z.y));
        tl = (double2)(0.0, AS_DOUBLE(AS_ULONG(r.lo) ^ (AS_ULONG(z.y) & SIGNBIT_DP64)));
    } else {
        double2 r = root2(fsub(sqr(x), 1.0));
        l2 = sqr(add(r, x));
        t = (double2)(BUILTIN_COPYSIGN_F64(r.hi, z.x), z.y);
        tl = (double2)(AS_DOUBLE(AS_ULONG(r.lo) ^ (AS_ULONG(z.x) & SIGNBIT_DP64)), 0.0);
    }

    double rr;
    if (b) {
        rr = 0.5 * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    double ri = MATH_PRIVATE(atan2ep)(con(t.y, tl.y), con(t.x, tl.x));

    if (!FINITE_ONLY_OPT()) {
        rr = (BUILTIN_ISINF_F64(z.x) | BUILTIN_ISINF_F64(z.y)) ? PINF_F64 : rr;
    }

    return (double2)(rr, ri);
}

