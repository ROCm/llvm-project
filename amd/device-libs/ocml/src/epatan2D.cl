/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double2
MATH_PRIVATE(epatan2)(double y, double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ay = BUILTIN_ABS_F64(y);
    double u = BUILTIN_MAX_F64(ax, ay);
    double v = BUILTIN_MIN_F64(ax, ay);

    // Lift u into [0.5,1) by an exact power of two (v scaled equally, v/u fixed)
    // so the divide's reconstruction word stays normal. Only scale up (v<=u).
    int eu = BUILTIN_FREXP_EXP_F64(u);
    if (eu < 0) {
        u = BUILTIN_FLDEXP_F64(u, -eu);
        v = BUILTIN_FLDEXP_F64(v, -eu);
    }
#ifdef EXTRA_ACCURACY
    double2 r = div2(v, u);
#else
    double2 r = div(v, u);
#endif

    // atan(r): secondary reduction by pi/6 (tp6=tan(pi/6), p6=pi/6) keeps the
    // poly argument |xr| <= tan(pi/12) = 0x1.126145e9ecd56p-2.
    const double2 tp6 = con(0x1.279a74590331cp-1, 0x1.34863e0792bedp-55);
    const double2 p6  = con(0x1.0c152382d7366p-1, -0x1.ee6913347c2a6p-55);
    int big = r.hi > 0x1.126145e9ecd56p-2;
    double2 xr = big ? div(sub(r, tp6), fadd(1.0, mul(tp6, r))) : r;
    double2 s = sqr(xr);
    double sh = s.hi;

    // Q(s) = 1/5 - s/7 + ...: high terms in double (PEn), low K terms in dd.
#ifdef EXTRA_ACCURACY
    double2 p = con(PE8(sh, 0x1.9a3a296aeecf4p-7, -0x1.6b412f701412dp-6, 0x1.aff9d707d48fep-6,
                            -0x1.d2a98dda816c1p-6, 0x1.f0577ca93f6d7p-6, -0x1.0840b321a9bbfp-5,
                            0x1.1a7b820febb25p-5, -0x1.2f684af6b5491p-5, 0x1.47ae14730908ep-5), 0.0);
    p = add(mul(p, s), con(-0x1.642c85907c632p-5, -0x1.7bd8b1532533p-62 ));
    p = add(mul(p, s), con( 0x1.861861861746fp-5,  0x1.e92e7a08abd21p-59));
    p = add(mul(p, s), con(-0x1.af286bca1aee2p-5,  0x1.b50d5152b110p-67 ));
    p = add(mul(p, s), con( 0x1.e1e1e1e1e1e1dp-5,  0x1.372d96d67fb51p-59));
    p = add(mul(p, s), con(-0x1.1111111111111p-4, -0x1.0337fabd5a09cp-60));
    p = add(mul(p, s), con( 0x1.3b13b13b13b14p-4, -0x1.3b18c0c2cdc61p-58));
    p = add(mul(p, s), con(-0x1.745d1745d1746p-4,  0x1.745d20c52ced6p-59));
    p = add(mul(p, s), con( 0x1.c71c71c71c71cp-4,  0x1.c71c71c48bebdp-58));
    p = add(mul(p, s), con(-0x1.2492492492492p-3, -0x1.2492492491f56p-57));
    p = add(mul(p, s), con( 0x1.999999999999ap-3, -0x1.999999999999ap-57));
#else
    double2 p = con(PE9(sh, -0x1.9a7f3ff7a4006p-6, 0x1.368915fb6fecfp-5, -0x1.626317ea15db4p-5,
                            0x1.85f9ad0bffb54p-5, -0x1.af270c9156633p-5, 0x1.e1e1d75df9076p-5,
                            -0x1.111110f6778cdp-4, 0x1.3b13b13abe39ap-4, -0x1.745d1745d0d22p-4,
                            0x1.c71c71c71c712p-4), 0.0);
    p = add(mul(p, s), con(-0x1.2492492492492p-3, -0x1.1ce994e1c4c9ap-57));
    p = add(mul(p, s), con( 0x1.999999999999ap-3, -0x1.999a168c46bf3p-57));
#endif

    // atan(xr) = xr + xr^3*(-1/3 + s*Q(s)), then undo the pi/6 reduction.
    const double2 c3 = con(-0x1.5555555555555p-2, -0x1.5555555555555p-56);
    double2 a = fadd(xr, mul(mul(s, xr), add(c3, mul(s, p))));
    a = big ? add(p6, a) : a;

    // Quadrant fold: (ax<ay) reflects across pi/4, x<0 across pi/2, sign from y.
    const double2 piby2 = con(0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54);
    const double2 pi    = con(0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53);
    a = ax < ay          ? sub(piby2, a) : a;
    a = AS_INT2(x).y < 0 ? sub(pi, a)    : a;
    return csgn(a, con(y, y));
}
