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

        float2 den = fadd(mul(d, y*qt), d);
        ret = MATH_DIV(n.hi, den.hi);
        ret = x == 0.0f ? BUILTIN_COPYSIGN_F32(PINF_F32, x) : ret;
        ret = x < 0.0f && BUILTIN_TRUNC_F32(x) == x ? QNAN_F32 : ret;
    } else {
        float xr = MATH_FAST_RCP(ax);
        float p = PE3(xr, -0x1.61f140p-9f, 0x1.c72f8cp-9f, 0x1.555554p-4f, 1.0f);

        float2 e = sub(mul(MATH_PRIVATE(epln)(ax), ax - 0.5f), ax);

        if (x > 0.0f) {
            const float sqrt2pi = 0x1.40d932p+1f;
            float m = MATH_PRIVATE(expep)(e);
            float g = sqrt2pi * m * p;
            ret = x > 0x1.18521ep+5f ? PINF_F32 : g;
        } else {
            const float2 lnsqrtpiby2 = con(0x1.ce6bb2p-3f, 0x1.6a84c6p-29f);
            float s = -x * MATH_MANGLE(sinpi)(x);
            float asp = BUILTIN_ABS_F32(s) * p;
            float mag = MATH_PRIVATE(expep)(sub(sub(lnsqrtpiby2, e), MATH_PRIVATE(epln)(asp)));
            ret = BUILTIN_COPYSIGN_F32(mag, s);
            ret = BUILTIN_TRUNC_F32(x) == x || BUILTIN_ISNAN_F32(x) ? QNAN_F32 : ret;
        }
    }

    return ret;
}
