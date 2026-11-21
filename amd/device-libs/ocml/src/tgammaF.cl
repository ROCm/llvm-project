/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epln)(float);
extern CONSTATTR float MATH_PRIVATE(expep)(float2);

CONSTATTR float
MATH_MANGLE(tgamma)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 16.0f) {
        float2 n, d;
        float y = x;
        if (x > 0.0f) {
            n = con(1.0f, 0.0f);
            while (y > 2.5f) {
                n = omul(n, y - 1.0f);
                y = y - 1.0f;
                n = omul(n, y - 1.0f);
                y = y - 1.0f;
            }
            if (y > 1.5f) {
                n = omul(n, y - 1.0f);
                y = y - 1.0f;
            }
            if (x >= 0.5f)
                y = y - 1.0f;
            d = con(x < 0.5f ? x : 1.0f, 0.0f);
        } else {
            d = con(x, 0.0f);
            while (y < -1.5f) {
                d = omul(d, y + 1.0f);
                y = y + 1.0f;
                d = omul(d, y + 1.0f);
                y = y + 1.0f;
            }
            if (y < -0.5f) {
                d = omul(d, y + 1.0f);
                y = y + 1.0f;
            }
            n = con(1.0f, 0.0f);
        }
        float qt = PE7(y, -0x1.1201dcp-10f, 0x1.d16868p-8f, -0x1.3c8284p-7f, -0x1.598558p-5f, 0x1.55148ep-3f,
                         -0x1.581830p-5f, -0x1.4fcf46p-1f, 0x1.2788d0p-1f);

        float2 den = fadd(mul(d, mul(y, qt)), d);

        // (n.hi + n.lo) / (den.hi + den.lo); scale a tiny den.hi so the reciprocal
        // sees a normal input
        float sc = ax < 0x1.0p-64f ? 0x1.0p+64f : 1.0f;
        float r = MATH_FAST_RCP(den.hi * sc) * sc;
        float q = n.hi * r;
        ret = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-q, den.lo, BUILTIN_FMA_F32(-q, den.hi, n.hi) + n.lo), r, q);
        ret = BUILTIN_ISINF_F32(q) ? q : ret;
        ret = x == 0.0f ? BUILTIN_COPYSIGN_F32(PINF_F32, x) : ret;
        ret = x < 0.0f && BUILTIN_TRUNC_F32(x) == x ? QNAN_F32 : ret;
    } else {
        // e = lgamma(ax) = (ax - 0.5)*ln(ax) - ax + ln(sqrt(2pi)) + c(1/ax)
        float xr = MATH_FAST_RCP(ax);
        float c = xr * PE2(xr*xr, 0x1.a01a02p-11f, -0x1.6c16c2p-9f, 0x1.555556p-4f);
        float2 e = fadd(fadd(sub(mul(MATH_PRIVATE(epln)(ax), ax - 0.5f), ax), con(0x1.d67f1cp-1f, 0x1.0c97d6p-26f)), c);

        if (x > 0.0f) {
            ret = MATH_PRIVATE(expep)(e);
            ret = x > 0x1.18521ep+5f ? PINF_F32 : ret;
        } else {
            // |Gamma(x)| = pi / (|x*sinpi(x)| * Gamma(ax)), computed 2^64 times larger
            // so that exp of the high part stays normal for a normal result
            float sp = MATH_MANGLE(sinpi)(x);
            float sh = ax * BUILTIN_ABS_F32(sp);
            float sl = BUILTIN_FMA_F32(ax, BUILTIN_ABS_F32(sp), -sh);
            float2 ls = add(MATH_PRIVATE(epln)(sh), sl * MATH_FAST_RCP(sh));
            float2 l = sub(sub(con(0x1.6c0c98p+5f, 0x1.4176fep-23f), e), ls);
            float mag = BUILTIN_FLDEXP_F32(MATH_PRIVATE(expep)(l), -64);
            ret = BUILTIN_COPYSIGN_F32(mag, sp);
            ret = BUILTIN_TRUNC_F32(x) == x || BUILTIN_ISNAN_F32(x) ? QNAN_F32 : ret;
        }
    }

    return ret;
}
