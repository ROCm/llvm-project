/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#if defined EXTRA_ACCURACY
extern CONSTATTR double2 MATH_PRIVATE(epln)(double);
#endif

CONSTATTR double
MATH_MANGLE(erfinv)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double t, q, c0h, c0l, m, ml;
#if defined EXTRA_ACCURACY
    double tl = 0.0;
#endif

    if (ax < 0.375) {
        t = ax*ax;
        q = PE12(t, -0x1.c8d1035433bc3p+1, 0x1.cae6f422d3a5fp+1, -0x1.8b97533105aeep+0, 0x1.b5f556b104eb9p-2,
                 -0x1.3f084fb04aedep-5, 0x1.4153ce2d5643fp-5, 0x1.25f062b6efecbp-5,
                 0x1.5f2c819c9a4c6p-5, 0x1.a7c603995125cp-5, 0x1.0a131f91d6f5p-4,
                 0x1.62847c2f2c688p-4, 0x1.053c2c0ad0048p-3, 0x1.db29fb2fee4d7p-3);
        c0h = 0x1.c5bf891b4ef6bp-1;
        c0l = -0x1.4212b3f40eb42p-55;
        m = ax;
        ml = 0.0;
    } else if (ax < 0x1.fffep-1) {
#if defined EXTRA_ACCURACY
        // w + wl = -log(1 - ax^2) with 1 - ax^2 = uh + ul exact
        double xx = ax * ax;
        double uh = 1.0 - xx;
        double ul = ((1.0 - uh) - xx) - BUILTIN_FMA_F64(ax, ax, -xx);
        double2 lu = MATH_PRIVATE(epln)(uh);
        double wl = -BUILTIN_FMA_F64(ul, BUILTIN_AMDGPU_RCP_F64(uh), lu.lo);
        double w = wl - lu.hi;
        wl = wl - (w + lu.hi);
#else
        double w = -MATH_MANGLE(log)(BUILTIN_FMA_F64(-ax, ax, 1.0));
#endif

        if (w < 6.25) {
            t = w - 3.125;
#if defined EXTRA_ACCURACY
            double tb = t - w;
            tl = ((w - (t - tb)) + (-3.125 - tb)) + wl;
#endif
            q = PE23(t, -0x1.c081a851e46b5p-72, 0x1.f2206a6bc0822p-69, 0x1.82bfff793ab7dp-66,
                     -0x1.a0511c81fe07ep-62, 0x1.62209c1adf1fbp-61, 0x1.204fc34a87219p-56,
                     -0x1.199060eb1148ep-53, -0x1.796454a496497p-55, 0x1.d5eb5a2d19887p-48,
                     -0x1.685bf01812e43p-45, -0x1.6b8b52692bd9ep-44, 0x1.723470b4b5892p-39,
                     -0x1.c8a670cae9079p-37, -0x1.dc3ec9519f46ep-35, 0x1.20f542fbfaafep-30,
                     -0x1.1a9eb220c5558p-28, -0x1.f36cee9443884p-26, 0x1.c6b4f891cd3fap-22,
                     -0x1.6e8a53c8dfd7p-20, -0x1.d1d1f7ca41c0ap-17, 0x1.879c2a20b022ep-13,
                     -0x1.8457694835449p-11, -0x1.8b6c33114ea59p-8, 0x1.ebd80d9b13dfap-3);
            c0h = 0x1.a755e7c99ae86p+0;
            c0l = -0x1.a02ef3976ece7p-59;
        } else {
            // t = sqrt(w) - 2.875 with the square root residual
            double s = MATH_FAST_SQRT(w);
#if defined EXTRA_ACCURACY
            t = (s - 2.875) + (BUILTIN_FMA_F64(-s, s, w) + wl) * (0.5 * BUILTIN_AMDGPU_RSQRT_F64(w));
#else
            t = (s - 2.875) + BUILTIN_FMA_F64(-s, s, w) * (0.5 * BUILTIN_AMDGPU_RSQRT_F64(w));
#endif
            q = PE14(t, 0x1.3ecb5687a1532p-19, -0x1.a6fced7c3ff47p-18, 0x1.fdc46d21fa07fp-21,
                     0x1.c85d59a55ba2fp-16, -0x1.005b0d3d02ed4p-14, 0x1.5a91a34d66715p-17,
                     0x1.e9edbbf74a513p-13, -0x1.196bcdd5e3508p-11, 0x1.0edb992a43996p-12,
                     0x1.7d47397a32149p-10, -0x1.32e54e5601b75p-8, 0x1.0eb822072b651p-7,
                     -0x1.6bc3164a67e87p-7, 0x1.a75c4b278353cp-7, 0x1.ff70ae3216d08p-1);
            c0h = 0x1.5a9d7f10825d9p+1;
            c0l = 0x1.9fc523a15507cp-53;
        }
        m = ax;
        ml = 0.0;
    } else {
        // m + ml = sqrt(-log(1 - ax))
#if defined EXTRA_ACCURACY
        double2 lv = MATH_PRIVATE(epln)(1.0 - ax);
        double l = -lv.hi;
        double ll = -lv.lo;
#else
        double l = -MATH_MANGLE(log)(1.0 - ax);
#endif
        m = MATH_FAST_SQRT(l);
        double r = MATH_FAST_RCP(m);
#if defined EXTRA_ACCURACY
        ml = (BUILTIN_FMA_F64(-m, m, l) + ll) * (0.5 * r);
#else
        ml = BUILTIN_FMA_F64(-m, m, l) * (0.5 * r);
#endif

        if (ax < 0x1.fffffffep-1) {
            t = r - 0.25;
            q = PE11(t, -0x1.17153bb07bd4ap+9, 0x1.0b72df425c375p+8, -0x1.7416d2c6d6bbap+6, 0x1.07d6a3ee26abcp+5,
                     -0x1.8bb8b9f8e85b1p+3, 0x1.38212bd1ad9acp+2, -0x1.03f017396277p+1,
                     0x1.f81957abb2942p-1, -0x1.67586351a22bep-1, 0x1.36b9de7a014abp-1,
                     -0x1.1889f5f773a38p-2, -0x1.8113f7498837p-2);
            c0h = 0x1.e02c8b0119e11p-1;
            c0l = -0x1.44c00dafd00e9p-55;
        } else {
            t = r - 0.1875;
            q = PE9(t, -0x1.bfff0bf2cbd98p+9, 0x1.f5eb1b9b9454p+7, -0x1.152d313f86278p+6, 0x1.4be6e64a56d89p+4,
                    -0x1.ab16dc5a845acp+2, 0x1.34ac3cdf23ebcp+1, -0x1.306009bff3f0bp+0,
                    0x1.ab1d6c4b1b7dp-1, -0x1.a0f84605fb969p-2, -0x1.55ef4d473b132p-2);
            c0h = 0x1.eb93ee7efcd57p-1;
            c0l = 0x1.8e6f0459770dcp-56;
        }
    }

    // ph + pl = c0 + t*q, then (ph + pl)*(m + ml) with one rounding
    double ph = BUILTIN_FMA_F64(t, q, c0h);
#if defined EXTRA_ACCURACY
    double pl = BUILTIN_FMA_F64(tl, q, BUILTIN_FMA_F64(t, q, c0h - ph) + c0l);
#else
    double pl = BUILTIN_FMA_F64(t, q, c0h - ph) + c0l;
#endif
    double ret = BUILTIN_FMA_F64(ph, m, BUILTIN_FMA_F64(ph, ml, pl * m));

    if (!FINITE_ONLY_OPT()) {
        ret = ax > 1.0 ? QNAN_F64 : ret;
        ret = ax == 1.0 ? PINF_F64 : ret;
    }

    return BUILTIN_COPYSIGN_F64(ret, x);
}

