/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

#if defined EXTRA_ACCURACY
#include "trigredF.h"
#endif

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 z);

CONSTATTR float2
MATH_MANGLE(ccosh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e430p+0, -0x1.05c610p-28f)));
    float2 er = rcp(e);
    er = ldx(er, -4);
    float2 cx = fadd(e, er);
    float2 sx = fsub(e, er);
#if defined EXTRA_ACCURACY
    float4 sc = MATH_PRIVATE(epsincos)(z.y);
    // Fold the final scaling by 2 into cos and sin so that the products
    // don't go denormal for a normal result
    bool s = x >= 0x1.0p-12f;
    float2 sxs = s ? sx : con(x, 0.0f);
    float2 cy = ldx(sc.lo, 1);

    // Also keep the low part of sinh(x)*sin(y) from underflowing for tiny y
    int ky = BUILTIN_ABS_F32(z.y) < 0x1.0p-100f ? 64 : 0;
    float2 sy = ldx(sc.hi, s + ky);
    float rr = BUILTIN_FMA_F32(cx.hi, cy.hi, BUILTIN_FMA_F32(cx.lo, cy.hi, cx.hi * cy.lo));
    float ri = BUILTIN_FMA_F32(sxs.hi, sy.hi, BUILTIN_FMA_F32(sxs.lo, sy.hi, sxs.hi * sy.lo));

    if (!FINITE_ONLY_OPT()) {
        bool b = x >= 0x1.686fc0p+6f;
        rr = b ? PINF_F32 * cy.hi : rr;
        ri = b ? PINF_F32 * sy.hi : ri;
    }

    ri = BUILTIN_FLDEXP_F32(AS_FLOAT(AS_UINT(ri) ^ (AS_UINT(z.x) & SIGNBIT_SP32)), -ky);

#else
    float cy;
    float sy = MATH_MANGLE(sincos)(z.y, &cy);

    float cxhi, sxhi;
    if (FINITE_ONLY_OPT()) {
        cxhi = cx.hi;
        sxhi = sx.hi;
    } else {
        bool b = x >= 0x1.686fc0p+6f;
        cxhi = b ? PINF_F32 : cx.hi;
        sxhi = b ? PINF_F32 : sx.hi;
    }

    float rr = BUILTIN_FLDEXP_F32(cxhi * cy, 1);
    bool s = x >= 0x1.0p-12f;
    float ri = BUILTIN_FLDEXP_F32(BUILTIN_COPYSIGN_F32(s ? sxhi : x, z.x) * sy, s);
    
#endif

    if (!FINITE_ONLY_OPT()) {
        ri = ((x == 0.0f) | (z.y == 0.0f)) ? BUILTIN_COPYSIGN_F32(0.0f, z.y) : ri;
        rr = (BUILTIN_ISINF_F32(x) &
              BUILTIN_CLASS_F32(z.y, CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER|CLASS_QNAN|CLASS_SNAN)) ? x : rr;
    }

    return (float2)(rr, ri);
}

