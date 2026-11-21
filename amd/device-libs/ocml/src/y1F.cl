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

extern float MATH_PRIVATE(sinb)(float, int, float);
extern CONSTATTR float MATH_PRIVATE(bp1)(float);
extern CONSTATTR float MATH_PRIVATE(ba1)(float);

CONSTATTR float
MATH_MANGLE(y1)(float x)
{
    const float b0 = 0.5f;
    const float b1 = 0.625f;
    const float b2 = 0.75f;
    const float b3 = 0.9375f;
    const float b4 = 1.21875f;
    const float b5 = 1.53125f;
    const float b6 = 1.84375f;
    const float b7 = 2.078125f;
    const float b8 = 2.3125f;
    const float b9 = 2.734375f;
    const float b10 = 3.15625f;
    const float b11 = 4.203125f;
    const float b12 = 4.6875f;
    const float b13 = 6.1875f;
    const float b14 = 7.76953125f;
    const float b15 = 9.359375f;
    const float b16 = 10.9375f;
    const float b17 = 12.5625f;

    float ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_Y1);
        float ch, cl, c0l, c1l;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0f;
                    cl = 0.0f;
                    c0l = 0.0f;
                    c1l = 0.0f;
                    p += 0*9;
                } else if (x < b1) {
                    ch = 0x1.0p-1f;
                    cl = 0.0f;
                    c0l = -0x1.cf4272p-28f;
                    c1l = 0.0f;
                    p += 1*9;
                } else if (x < b2) {
                    ch = 0x1.4p-1f;
                    cl = 0.0f;
                    c0l = -0x1.de33dp-28f;
                    c1l = 0.0f;
                    p += 2*9;
                } else if (x < b3) {
                    ch = 0x1.8p-1f;
                    cl = 0.0f;
                    c0l = 0x1.9cff72p-28f;
                    c1l = 0.0f;
                    p += 3*9;
                } else {
                    ch = 0x1.ep-1f;
                    cl = 0.0f;
                    c0l = 0x1.5a0b1ep-28f;
                    c1l = 0.0f;
                    p += 4*9;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.38p+0f;
                    cl = 0.0f;
                    c0l = -0x1.500dbcp-26f;
                    c1l = 0.0f;
                    p += 5*9;
                } else if (x < b6) {
                    ch = 0x1.88p+0f;
                    cl = 0.0f;
                    c0l = 0x1.71327ep-27f;
                    c1l = 0.0f;
                    p += 6*9;
                } else if (x < b7) {
                    ch = 0x1.d8p+0f;
                    cl = 0.0f;
                    c0l = -0x1.0d40dep-28f;
                    c1l = 0.0f;
                    p += 7*9;
                } else {
                    ch = 0x1.193beep+1f;
                    cl = -0x1.6401b8p-24f;
                    c0l = 0.0f;
                    c1l = 0x1.0a6684p-27f;
                    p += 8*9;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.28p+1f;
                    cl = 0.0f;
                    c0l = 0x1.034746p-31f;
                    c1l = 0.0f;
                    p += 9*9;
                } else if (x < b10) {
                    ch = 0x1.5ep+1f;
                    cl = 0.0f;
                    c0l = 0x1.68794ap-32f;
                    c1l = 0.0f;
                    p += 10*9;
                } else if (x < b11) {
                    ch = 0x1.d76d4ap+1f;
                    cl = 0x1.ff742ep-24f;
                    c0l = 0x1.3f552ep-32f;
                    c1l = 0.0f;
                    p += 11*9;
                } else if (x < b12) {
                    ch = 0x1.0dp+2f;
                    cl = 0.0f;
                    c0l = -0x1.96204cp-27f;
                    c1l = 0.0f;
                    p += 12*9;
                } else {
                    ch = 0x1.5b7fe4p+2f;
                    cl = 0x1.d0f606p-23f;
                    c0l = 0.0f;
                    c1l = 0x1.29f76ep-27f;
                    p += 13*9;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.bc418ap+2f;
                    cl = -0x1.f4ef56p-23f;
                    c0l = 0x1.2eeb4ep-27f;
                    c1l = 0.0f;
                    p += 14*9;
                } else if (x < b15) {
                    ch = 0x1.13127ap+3f;
                    cl = 0x1.cc2d36p-22f;
                    c0l = 0.0f;
                    c1l = -0x1.13008ep-32f;
                    p += 15*9;
                } else if (x < b16) {
                    ch = 0x1.43f2eep+3f;
                    cl = 0x1.47a32p-23f;
                    c0l = -0x1.a39e62p-27f;
                    c1l = 0.0f;
                    p += 16*9;
                } else {
                    ch = 0x1.77f914p+3f;
                    cl = -0x1.caf37ep-23f;
                    c0l = 0.0f;
                    c1l = 0x1.3df79p-31f;
                    p += 17*9;
                }
            }
        }

        float x2 = x*x;
        // t + tl = (x < b0 ? x2 : x) - ch - cl; split the last Horner step
        float th = (x < b0 ? x2 : x) - ch;
        float t = th - cl;
        float tl = (th - t) - cl;
        float s = PE6(t, p[8], p[7], p[6], p[5], p[4], p[3], p[2]);
        float rh = BUILTIN_FMA_F32(t, s, p[1]);
        float d = p[1] - rh;
        float rl = BUILTIN_FMA_F32(t, s, d) + (p[1] - (d + rh));

        // p[0] + t*rh exactly, then the small terms with a single final rounding
        float ph = t * rh;
        float pl = BUILTIN_FMA_F32(t, rh, -ph);
        float a = p[0] + ph;
        float bb = a - p[0];
        float er = ((p[0] - (a - bb)) + (ph - bb)) + pl;
        float e = BUILTIN_FMA_F32(t, rl + c1l, er + BUILTIN_FMA_F32(tl, BUILTIN_FMA_F32(2.0f * t, p[2], p[1]), c0l));
        ret = a + e;

        if (x < b0) {
            // (2/pi)(j1(x) log(x) - 1/x) + x*(a + e) with a single final rounding
            float jm = x2 * PE3(x2, 0x1.6c16c2p-20f, -0x1.c71c72p-14f, 0x1.555556p-8f, -0x1.0p-3f);
            float jh = 1.0f + jm;
            float2 j = mul(0.5f * x, con(jh, (1.0f - jh) + jm));
            float q = MATH_FAST_RCP(x);
            float2 r = fadd(q, q * BUILTIN_FMA_F32(-x, q, 1.0f));
            float2 v = mul(con(0x1.45f306p-1f, 0x1.b93910p-26f), sub(mul(j, MATH_PRIVATE(epln)(x)), r));
            float xa = x * a;
            float sh = v.hi + xa;
            float sb = sh - v.hi;
            float sl = ((v.hi - (sh - sb)) + (xa - sb)) + (v.lo + BUILTIN_FMA_F32(x, e, BUILTIN_FMA_F32(x, a, -xa)));
            ret = sh + sl;

            // subnormal x: -(2/pi)/x, scaled so the reciprocal sees a normal input
            float xs = BUILTIN_ABS_F32(x) * 0x1.0p+32f;
            float rs = MATH_FAST_RCP(xs);
            float qs = -0x1.45f306p-1f * rs;
            qs = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-qs, xs, -0x1.45f306p-1f) - 0x1.b93910p-26f, rs, qs);
            qs = xs == 0.0f ? NINF_F32 : qs * 0x1.0p+32f;
            ret = x < 0x1.0p-126f ? qs : ret;
            ret = x < 0.0f ? QNAN_F32 : ret;
        }
    } else {
        float r = MATH_RCP(x);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp1)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_AMDGPU_RSQRT_F32(x) * MATH_PRIVATE(ba1)(r2) * MATH_PRIVATE(sinb)(x, 1, p);
        ret = x == PINF_F32 ? 0.0f : ret;
    }

    return ret;
}

