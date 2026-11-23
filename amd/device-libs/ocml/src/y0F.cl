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
extern CONSTATTR float MATH_PRIVATE(bp0)(float);
extern CONSTATTR float MATH_PRIVATE(ba0)(float);

CONSTATTR float
MATH_MANGLE(y0)(float x)
{
    const float b0  = 0.3125f;
    const float b1  = 0.4375f;
    const float b2  = 0.5625f;
    const float b3  = 0.6875f;
    const float b4  = 0.8125f;
    const float b5  = 1.0f;
    const float b6  = 1.25f;
    const float b7  = 1.625f;
    const float b8  = 2.0f;
    const float b9  = 2.53125f;
    const float b10 = 3.0f;
    const float b11 = 3.484375f;
    const float b12 = 4.703125f;
    const float b13 = 6.265625f;
    const float b14 = 7.84375f;
    const float b15 = 9.421875f;
    const float b16 = 10.984375f;
    const float b17 = 12.546875f;

    float ret;
    float xin = x;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(float, p, M32_Y0);
        float ch, cl, c0l, c1l;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0f;
                    cl = 0.0f;
                    c0l = 0.0f;
                    c1l = 0.0f;
                } else if (x < b1) {
                    ch = 0x1.4p-2f;
                    cl = 0.0f;
                    c0l = -0x1.8dcd86p-26f;
                    c1l = 0.0f;
                    p += 1*9;
                } else if (x < b2) {
                    ch = 0x1.cp-2f;
                    cl = 0.0f;
                    c0l = -0x1.32c796p-26f;
                    c1l = 0.0f;
                    p += 2*9;
                } else if (x < b3) {
                    ch = 0x1.2p-1f;
                    cl = 0.0f;
                    c0l = 0x1.1b99c4p-27f;
                    c1l = 0.0f;
                    p += 3*9;
                } else {
                    ch = 0x1.6p-1f;
                    cl = 0.0f;
                    c0l = 0x1.1c934ep-29f;
                    c1l = 0.0f;
                    p += 4*9;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.c982ecp-1f;
                    cl = -0x1.cafa06p-27f;
                    c0l = 0.0f;
                    c1l = -0x1.f24424p-27f;
                    p += 5*9;
                } else if (x < b6) {
                    ch = 0x1.p+0f;
                    cl = 0.0f;
                    c0l = 0x1.52fbeep-29f;
                    c1l = 0.0f;
                    p += 6*9;
                } else if (x < b7) {
                    ch = 0x1.4p+0f;
                    cl = 0.0f;
                    c0l = -0x1.508f2ap-27f;
                    c1l = 0.0f;
                    p += 7*9;
                } else {
                    ch = 0x1.ap+0f;
                    cl = 0.0f;
                    c0l = 0x1.3988dap-28f;
                    c1l = 0.0f;
                    p += 8*9;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.193beep+1f;
                    cl = -0x1.6401b8p-24f;
                    c0l = 0x1.06f3bep-27f;
                    c1l = 0.0f;
                    p += 9*9;
                } else if (x < b10) {
                    ch = 0x1.44p+1f;
                    cl = 0.0f;
                    c0l = -0x1.e35736p-29f;
                    c1l = 0.0f;
                    p += 10*9;
                } else if (x < b11) {
                    ch = 0x1.8p+1f;
                    cl = 0.0f;
                    c0l = 0x1.09ad62p-29f;
                    c1l = 0.0f;
                    p += 11*9;
                } else if (x < b12) {
                    ch = 0x1.fa9534p+1f;
                    cl = 0x1.b30ad4p-24f;
                    c0l = 0.0f;
                    c1l = 0x1.3e942ep-27f;
                    p += 12*9;
                } else {
                    ch = 0x1.5b7fe4p+2f;
                    cl = 0x1.d0f606p-23f;
                    c0l = 0x1.495c66p-27f;
                    c1l = 0.0f;
                    p += 13*9;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.c581dcp+2f;
                    cl = 0x1.39c84p-24f;
                    c0l = 0.0f;
                    c1l = -0x1.7a53bp-32f;
                    p += 14*9;
                } else if (x < b15) {
                    ch = 0x1.13127ap+3f;
                    cl = 0x1.cc2d36p-22f;
                    c0l = 0x1.84e164p-30f;
                    c1l = 0.0f;
                    p += 15*9;
                } else if (x < b16) {
                    ch = 0x1.471d74p+3f;
                    cl = -0x1.4b7056p-22f;
                    c0l = 0.0f;
                    c1l = -0x1.847a3p-28f;
                    p += 16*9;
                } else {
                    ch = 0x1.77f914p+3f;
                    cl = -0x1.caf37ep-23f;
                    c0l = -0x1.752a2ap-30f;
                    c1l = 0.0f;
                    p += 17*9;
                }
            }
        }

        // ret + retl = (2/pi) j0(x) log(x) for small x
        float retl = 0.0f;
        ret = 0.0f;
        if (x < b0) {
            float x2 = x * x;
            float jm = x2 * PE3(x2, 0x1.c71c72p-18f, -0x1.c71c72p-12f, 0x1.0p-6f, -0x1.0p-2f);
            float jh = 1.0f + jm;
            float2 v = mul(mul(con(0x1.45f306p-1f, 0x1.b93910p-26f), MATH_PRIVATE(epln)(x)),
                           con(jh, (1.0f - jh) + jm));
            ret = v.hi;
            retl = v.lo;
            x = x2;
        }

        // t + tl = x - ch - cl; split the last Horner step
        float th = x - ch;
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
        float sh = ret + a;
        float sb = sh - ret;
        ret = sh + ((((ret - (sh - sb)) + (a - sb)) + retl) + e);

        if (!FINITE_ONLY_OPT()) {
            ret = xin < 0.0f ? QNAN_F32 : ret;
            ret = xin == 0.0f ? NINF_F32 : ret;
        }
    } else {
        float r = MATH_RCP(x);
        float r2 = r*r;
        float p = MATH_PRIVATE(bp0)(r2) * r;
        ret = 0x1.988454p-1f * BUILTIN_AMDGPU_RSQRT_F32(x) * MATH_PRIVATE(ba0)(r2) * MATH_PRIVATE(sinb)(x, 0, p);
        ret = x == PINF_F32 ? 0.0f : ret;
    }

    return ret;
}

