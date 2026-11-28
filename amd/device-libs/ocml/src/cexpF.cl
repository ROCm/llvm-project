/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined EXTRA_ACCURACY
#define FLOAT_SPECIALIZATION
#include "ep.h"
#include "trigredF.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2);
#endif

CONSTATTR float2
MATH_MANGLE(cexp)(float2 z)
{
    float x = z.s0;
    float y = z.s1;
#if defined EXTRA_ACCURACY
    // e^x = 2^n e^t with e^t, cos(y) and sin(y) as hi + lo and one rounding
    // per component before the 2^n; the clamp keeps the reduction finite and
    // lets NaN through
    float4 sc = MATH_PRIVATE(epsincos)(y);
    float2 c = sc.lo;
    float2 sn = sc.hi;
    float xc = x < -110.0f ? -110.0f : x;
    xc = xc > 90.0f ? 90.0f : xc;
    float n = BUILTIN_RINT_F32(xc * 0x1.715476p+0f);
    // n*ln2_hi is exact, and so is xc - n*ln2_hi
    float2 e = MATH_PRIVATE(epexpep)(add(BUILTIN_FMA_F32(-n, 0x1.62e400p-1f, xc), -n * 0x1.7f7d1cp-20f));
    int in = (int)n;
    float rr = BUILTIN_FLDEXP_F32(BUILTIN_FMA_F32(e.hi, c.hi, BUILTIN_FMA_F32(e.lo, c.hi, e.hi * c.lo)), in);
    // sin(y) is subnormal for subnormal y while the result may be normal
    float sh = sn.hi * 0x1.0p+32f;
    float sl = sn.lo * 0x1.0p+32f;
    float ri = BUILTIN_FLDEXP_F32(BUILTIN_FMA_F32(e.hi, sh, BUILTIN_FMA_F32(e.lo, sh, e.hi * sl)), in - 32);
#else
    float cy;
    float sy = MATH_MANGLE(sincos)(y, &cy);
    // e^x = 2^n 2^f with 2^n applied last, so nothing overflows early; the
    // clamp keeps the reduction finite and lets NaN through
    float xc = x < -110.0f ? -110.0f : x;
    xc = xc > 90.0f ? 90.0f : xc;
    float ph = xc * 0x1.715476p+0f;
    float pl = BUILTIN_FMA_F32(xc, 0x1.4ae0c0p-26f, BUILTIN_FMA_F32(xc, 0x1.715476p+0f, -ph));
    float n = BUILTIN_RINT_F32(ph);
    float ef = BUILTIN_AMDGPU_EXP2_F32((ph - n) + pl);
    int in = (int)n;
    float rr = BUILTIN_FLDEXP_F32(ef * cy, in);
    // sin(y) is subnormal for subnormal y while the result may be normal
    float ri = BUILTIN_FLDEXP_F32(ef * (sy * 0x1.0p+32f), in - 32);
#endif

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F32(y);
        if (x == NINF_F32) {
            rr = 0.0f;
            ri = finite ? ri : 0.0f;
        }
        if (x == PINF_F32) {
            rr = finite ? rr : PINF_F32;
            ri = finite ? ri : QNAN_F32;
        }
        ri = y == 0.0f ? y : ri;
    }

    return (float2)(rr, ri);
}

