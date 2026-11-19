/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float2
MATH_PRIVATE(epatan2)(float y, float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    float u = BUILTIN_MAX_F32(ax, ay);
    float v = BUILTIN_MIN_F32(ax, ay);

    // Lift u into [0.5,1) by an exact power of two (v scaled equally, v/u fixed)
    // so the divide's reconstruction word stays normal. Only scale up (v<=u).
    int eu = BUILTIN_FREXP_EXP_F32(u);
    if (eu < 0) {
        u = BUILTIN_FLDEXP_F32(u, -eu);
        v = BUILTIN_FLDEXP_F32(v, -eu);
    }
#ifdef EXTRA_ACCURACY
    float2 r = div2(v, u);
#else
    float2 r = div(v, u);
#endif

    // atan(r): secondary reduction by pi/6 (tp6=tan(pi/6), p6=pi/6) keeps the
    // poly argument |xr| <= tan(pi/12) = 0x1.126146p-2f.
    const float2 tp6 = con(0x1.279a74p-1f, 0x1.640cc8p-27f);
    const float2 p6  = con(0x1.0c1524p-1f, -0x1.f4a326p-27f);
    int big = r.hi > 0x1.126146p-2f;
    float2 xr = big ? div(sub(r, tp6), fadd(1.0f, mul(tp6, r))) : r;
    float2 s = sqr(xr);
    float sh = s.hi;

    // Q(s) = 1/5 - s/7 + ...: high terms in float (PEn), low K terms in ff.
#ifdef EXTRA_ACCURACY
    float2 p = con(PE4(sh, -0x1.1acc44p-5f, 0x1.807ae4p-5f, -0x1.af3852p-5f,
                           0x1.e1eb88p-5f, -0x1.111168p-4f), 0.0f);
    p = add(mul(p, s), con( 0x1.3b13b4p-4f,  0x1.da6170p-31f));
    p = add(mul(p, s), con(-0x1.745d18p-4f,  0x1.579bc8p-29f));
    p = add(mul(p, s), con( 0x1.c71c72p-4f, -0x1.c5fd54p-31f));
    p = add(mul(p, s), con(-0x1.24924ap-3f,  0x1.b6db46p-28f));
    p = add(mul(p, s), con( 0x1.99999ap-3f, -0x1.99999ap-29f));
#else
    float2 p = con(PE5(sh, 0x1.8308aep-5f, -0x1.0c97c2p-4f, 0x1.3adcdcp-4f,
                           -0x1.745bbap-4f, 0x1.c71c6ep-4f, -0x1.24924ap-3f), 0.0f);
    p = add(mul(p, s), con(0x1.99999ap-3f, -0x1.999b5ep-29f));
#endif

    // atan(xr) = xr + xr^3*(-1/3 + s*Q(s)), then undo the pi/6 reduction.
    const float2 c3 = con(-0x1.555556p-2f, 0x1.555556p-27f);
    float2 a = fadd(xr, mul(mul(s, xr), add(c3, mul(s, p))));
    a = big ? add(p6, a) : a;

    // Quadrant fold: (ax<ay) reflects across pi/4, x<0 across pi/2, sign from y.
    const float2 piby2 = con(0x1.921fb6p+0f, -0x1.777a5cp-25f);
    const float2 pi    = con(0x1.921fb6p+1f, -0x1.777a5cp-24f);
    a = ax < ay        ? sub(piby2, a) : a;
    a = AS_INT(x) < 0  ? sub(pi, a)    : a;
    return csgn(a, con(y, y));
}
