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
MATH_PRIVATE(epln)(float a)
{
    int a_exp;
    float m = BUILTIN_FREXP_F32(a, &a_exp);
    int b = m < (2.0f/3.0f);
    m = BUILTIN_FLDEXP_F32(m, b);
    int e = a_exp - b;


    // x = (m - 1)/(m + 1): m - 1 is exact, and 1 + m is a Fast2Sum since
    // 1 has the larger (or equal) exponent
    float dh = 1.0f + m;
    float dl = (1.0f - dh) + m;
    float nh = m - 1.0f;
    float rc = MATH_FAST_RCP(dh);
    float xh = nh * rc;
    float xl = BUILTIN_FMA_F32(-xh, dl, BUILTIN_FMA_F32(-xh, dh, nh)) * rc;

    // s = x^2, q = c3 + s*p, c = x^3, t = c*q, each as hi + lo
    float sh = xh * xh;
    float sl = BUILTIN_FMA_F32(2.0f * xh, xl, BUILTIN_FMA_F32(xh, xh, -sh));
    float p = PE2(sh, 0x1.ed89c2p-3f, 0x1.23e988p-2f, 0x1.999bdep-2f);
    float qh = BUILTIN_FMA_F32(sh, p, 0x1.555554p-1f);
    float ql = BUILTIN_FMA_F32(sh, p, 0x1.555554p-1f - qh) + BUILTIN_FMA_F32(sl, p, 0x1.e72020p-29f);
    float ch = xh * sh;
    float cl = BUILTIN_FMA_F32(xh, sh, -ch) + BUILTIN_FMA_F32(xh, sl, xl * sh);
    float th = ch * qh;
    float tl = BUILTIN_FMA_F32(ch, qh, -th) + BUILTIN_FMA_F32(ch, ql, cl * qh);

    // ln(2)*e + 2*x + t: e*ln2_hi is exact, and each Fast2Sum adds the smaller term
    float fe = (float)e;
    float eh = fe * 0x1.62e400p-1f;
    float x2 = 2.0f * xh;
    float ah = eh + x2;
    float al = x2 - (ah - eh);
    float bh = ah + th;
    float bl = th - (bh - ah);
    float lo = ((al + bl) + BUILTIN_FMA_F32(fe, 0x1.7f7d1cp-20f, 2.0f * xl)) + tl;
    float rh = bh + lo;
    float2 r = con(rh, lo - (rh - bh));

    return r;
}

