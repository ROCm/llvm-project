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
MATH_MANGLE(casinh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double y = BUILTIN_ABS_F64(z.y);

    double2 l2, t;
    double2 tl = (double2)0.0;
    int e = 0;
    int kt = 0;
    bool b = true;

    bool big = x >= 0x1.0p+54 || y >= 0x1.0p+54;

    // The near axis formula below is accurate for x < 2^-27 * (1 - y^2)
    if (!big && (y >= 1.0 || x > 0x1.0p-28 * BUILTIN_FMA_F64(-y, y, 1.0))) {
        // For tiny y the imaginary part is linear in y and y^2 is negligible,
        // so scale y up to keep the intermediates normal
        kt = y < 0x1.0p-900 ? 512 : 0;
        double ys = BUILTIN_FLDEXP_F64(y, kt);
        double4 z2p1 = (double4)(add(mul(add(x,ys), sub(x,ys)), 1.0), mul(ys,x)*2.0);
        double4 rz2p1 = MATH_PRIVATE(epcsqrtep)(z2p1);
        double4 s = (double4)(add(rz2p1.lo, x), add(rz2p1.hi, ys));
        l2 = add(sqr(s.lo), sqr(s.hi));
        t = (double2)(s.s1, s.s3);
        tl = (double2)(s.s0, s.s2);
    } else if (big) {
        t = (double2)(x, y);
        e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(x, y));
        x = BUILTIN_FLDEXP_F64(x, -e);
        y = BUILTIN_FLDEXP_F64(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
    } else {
        b = false;
        double2 r = root2(fsub(1.0, sqr(y)));
        // Scale a tiny x up so that the residual stays normal
        int kn = x < 0x1.0p-900 ? 512 : 0;
        double ns = BUILTIN_FLDEXP_F64(x, kn);
        l2 = con(BUILTIN_FLDEXP_F64(divh(ns, r), -kn), 0.0);
        t = (double2)(r.hi, y);
        tl = (double2)(r.lo, 0.0);
    }

    double rr;
    if (b) {
        rr = 0.5 * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    rr = BUILTIN_COPYSIGN_F64(rr, z.x);
    double ri = BUILTIN_COPYSIGN_F64(BUILTIN_FLDEXP_F64(MATH_PRIVATE(atan2ep)(con(t.y, tl.y), con(t.x, tl.x)), -kt), z.y);

    if (!FINITE_ONLY_OPT()) {
        double i = BUILTIN_COPYSIGN_F64(PINF_F64, z.x);
        rr = (BUILTIN_ISINF_F64(z.x) | BUILTIN_ISINF_F64(z.y)) ? i : rr;
        ri = (z.y == 0.0) ? z.y : ri;
    }

    return (double2)(rr, ri);
}

