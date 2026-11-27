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
MATH_MANGLE(erfcinv)(double y)
{
    double ret;

    if (y > 0.625) {
        ret = MATH_MANGLE(erfinv)(1.0 - y);
    } else if (y > 0x1.0p-10) {
#if defined EXTRA_ACCURACY
        // w + wl = -log(y*(2 - y)) with y*(2 - y) = uh + ul exact
        double uh = BUILTIN_FMA_F64(-y, y, 2.0 * y);
        double ul = BUILTIN_FMA_F64(-y, y, 2.0 * y - uh);
        double2 lu = MATH_PRIVATE(epln)(uh);
        double wl = -BUILTIN_FMA_F64(ul, BUILTIN_AMDGPU_RCP_F64(uh), lu.lo);
        double w = wl - lu.hi;
        wl = wl - (w + lu.hi);
        double t = w - 3.125;
        double tb = t - w;
        double tl = ((w - (t - tb)) + (-3.125 - tb)) + wl;
#else
        double t = -MATH_MANGLE(log)(BUILTIN_FMA_F64(-y, y, 2.0 * y)) - 3.125;
#endif
        double q, c0h, c0l;
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
        double ph = BUILTIN_FMA_F64(t, q, c0h);
#if defined EXTRA_ACCURACY
        double pl = BUILTIN_FMA_F64(tl, q, BUILTIN_FMA_F64(t, q, c0h - ph) + c0l);
#else
        double pl = BUILTIN_FMA_F64(t, q, c0h - ph) + c0l;
#endif

        // (ph + pl) * (1 - y) with 1 - y = x + xl exact
        double x = 1.0 - y;
        double xl = (1.0 - x) - y;
        ret = BUILTIN_FMA_F64(ph, x, BUILTIN_FMA_F64(ph, xl, pl * x));
    } else {
        // s + sl = sqrt(-log(y))
#if defined EXTRA_ACCURACY
        double2 lv = MATH_PRIVATE(epln)(y);
        double l = -lv.hi;
#else
        double l = -MATH_MANGLE(log)(y);
#endif
        double s = MATH_FAST_SQRT(l);
        double r = MATH_FAST_RCP(s);
#if defined EXTRA_ACCURACY
        double sl = (BUILTIN_FMA_F64(-s, s, l) - lv.lo) * (0.5 * r);
#else
        double sl = BUILTIN_FMA_F64(-s, s, l) * (0.5 * r);
#endif

        double t, q, c0h, c0l;
        if (y > 0x1.0p-19) {
            t = r - 0.328125;
            q = PE11(t, -0x1.c59f7db0f1854p+5, 0x1.85a184a298cb7p+4, -0x1.4ff9b26084f9cp+3, 0x1.3bcfb21042fa7p+2,
                     -0x1.33ef20786960dp+1, 0x1.33fdb1e6e0874p+0, -0x1.4474c28014196p-1,
                     0x1.bd1f090338119p-2, -0x1.c7dcc5a99319bp-2, 0x1.bb810bcd5fc2ep-2,
                     -0x1.3aa2f6087af93p-3, -0x1.a2bef2f8bdc22p-2);
            c0h = 0x1.d0696b8788649p-1;
            c0l = -0x1.e0ffcbd3ee3cp-56;
        } else if (y > 0x1.0p-40) {
            t = r - 0.234375;
            q = PE11(t, -0x1.b9477d77c8165p+10, 0x1.faed9d3ad438ap+8, -0x1.3222534599734p+7, 0x1.9d1b4d6d5d82dp+5,
                     -0x1.235913abba3d6p+4, 0x1.b0b0c1132c501p+2, -0x1.548dc228381fep+1,
                     0x1.33ef4971d1b8cp+0, -0x1.92e5cb04e32aep-1, 0x1.4e802fc193627p-1,
                     -0x1.36c33e36c82ap-2, -0x1.77d83eb8b6015p-2);
            c0h = 0x1.e3259f8209489p-1;
            c0l = 0x1.5e97ae4ce0cabp-56;
        } else if (y > 0x1.0p-82) {
            t = r - 0.15625;
            q = PE11(t, -0x1.0c1f442278f75p+16, 0x1.24be0d4d7224fp+14, -0x1.eff29712633c2p+11,
                     0x1.b7471d0162ed5p+9, -0x1.9c9375555b779p+7, 0x1.9d02abdad4904p+5,
                     -0x1.bf76fe4508bedp+3, 0x1.0fac965cbd2a2p+2, -0x1.b1026a5a2e6fp+0,
                     0x1.02e37b0b352dfp+0, -0x1.f90df1558b796p-2, -0x1.3935b2aff4acfp-2);
            c0h = 0x1.f0b40dbff82ffp-1;
            c0l = -0x1.d29441e53a542p-55;
        } else if (y > 0x1.0p-200) {
            t = r - 0.109375;
            q = PE11(t, -0x1.bde3050f6e4e2p+21, 0x1.e2c765a3ccd5dp+18, -0x1.0764a8e971d02p+16,
                     0x1.4b916e1aa02bbp+13, -0x1.b50913e98cbep+10, 0x1.325f47e1f2525p+8,
                     -0x1.d4bca84843f14p+5, 0x1.94e4ce076da0ep+3, -0x1.b87eedb6ed708p+1,
                     0x1.7684aebf737dfp+0, -0x1.53ae8caecb32ep-1, -0x1.0231180c78714p-2);
            c0h = 0x1.f76e64bb14646p-1;
            c0l = -0x1.2b0fb6d1afac4p-55;
        } else if (y > 0x1.0p-400) {
            t = r - 0.0703125;
            q = PE10(t, 0x1.2860ad0ff2135p+24, -0x1.22a52f8456c4cp+21, 0x1.cc2dfe97087fbp+17,
                     -0x1.81f01efd69766p+14, 0x1.5bb96b87fde21p+11, -0x1.56b95e3e4de61p+8,
                     0x1.808a21714cfb6p+5, -0x1.0c50a148dce76p+3, 0x1.26b4f740670ddp+1,
                     -0x1.c0edaa6026a45p-1, -0x1.8a72fac61e123p-3);
            c0h = 0x1.fbe7ffca9c0c9p-1;
            c0l = 0x1.c707ec455359p-55;
        } else if (y > 0x1.0p-900) {
            t = r - 0.05078125;
            q = PE10(t, 0x1.099a249ae8169p+29, -0x1.fb4ff1238e6d7p+24, 0x1.13d3f77242f5ap+21,
                     -0x1.533901352a5a4p+17, 0x1.b9d6055513eccp+13, -0x1.3a6eaced9f88cp+10,
                     0x1.fef11e7aae393p+6, -0x1.023fb37c47b95p+4, 0x1.9b12adaeedf14p+1,
                     -0x1.0913b001f4c6ap+0, -0x1.3e4f487c845c2p-3);
            c0h = 0x1.fda6ca6b8e078p-1;
            c0l = -0x1.f0fc436306cdbp-56;
        } else {
            t = r - 0.0390625;
            q = PE6(t, 0x1.c27bbcb5cea49p+15, -0x1.bf83c0cf778d3p+11, 0x1.188a1602b846fp+8, -0x1.b54133d9b6b69p+4,
                    0x1.0c8d4beba8ee6p+2, -0x1.2a05f5d53d425p+0, -0x1.09a814c42a666p-3);
            c0h = 0x1.fe8229ca19f5ep-1;
            c0l = 0x1.80c711fb5662dp-55;
        }

        double ph = BUILTIN_FMA_F64(t, q, c0h);
        double pl = BUILTIN_FMA_F64(t, q, c0h - ph) + c0l;
        ret = BUILTIN_FMA_F64(s, ph, BUILTIN_FMA_F64(s, pl, sl * ph));
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ((y < 0.0) | (y > 2.0)) ? QNAN_F64 : ret;
        ret = y == 0.0 ? PINF_F64 : ret;
        ret = y == 2.0 ? NINF_F64 : ret;
    }

    return ret;
}

