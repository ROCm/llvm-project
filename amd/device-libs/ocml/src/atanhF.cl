/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"


CONSTATTR float
MATH_MANGLE(atanh)(float x)
{
    float y = BUILTIN_ABS_F32(x);
    // (1 + y)/(1 - y) = 2^e (1 + r)/(1 - r) with |r| <= 1/5, so that
    // atanh(y) = e*ln(2)/2 + atanh(r)
    int ae;
    float am = BUILTIN_FREXP_F32((1.0f + y) * MATH_FAST_RCP(1.0f - y), &ae);
    int e = ae - (am < (2.0f/3.0f));
    float uh = 1.0f + y;
    float ul = (1.0f - uh) + y;
    float vh = 1.0f - y;
    float vl = BUILTIN_FLDEXP_F32((1.0f - vh) - y, e);
    vh = BUILTIN_FLDEXP_F32(vh, e);

    // r = (u - v)/(u + v); u - v is exact in the high part
    float nh = uh - vh;
    float nl = ul - vl;
    float dh = uh + vh;
    float db = dh - uh;
    float dl = ((uh - (dh - db)) + (vh - db)) + (ul + vl);
    float rc = MATH_FAST_RCP(dh);
    float q = nh * rc;
    float2 r = fadd(q, BUILTIN_FMA_F32(-q, dl, BUILTIN_FMA_F32(-q, dh, nh) + nl) * rc);

    // atanh(r) = r + r^3 p(r^2)
    float s = r.hi * r.hi;
    float p = PE2(s, 0x1.36db58p-3f, 0x1.992b46p-3f, 0x1.5555b4p-2f);
    float z = add(mul(con(0x1.62e430p-2f, -0x1.05c610p-30f), (float)e), fadd(r, s * r.hi * p)).hi;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0f ? QNAN_F32 : z;
        z = y == 1.0f ? PINF_F32 : z;
    }

    return BUILTIN_COPYSIGN_F32(z, x);
}

