/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(epln)(double);

extern double MATH_PRIVATE(sinb)(double, int, double);
extern CONSTATTR double MATH_PRIVATE(bp1)(double);
extern CONSTATTR double MATH_PRIVATE(ba1)(double);

CONSTATTR double
MATH_MANGLE(y1)(double x)
{
    const double b0 = 0.5;
    const double b1 = 0.625;
    const double b2 = 0.75;
    const double b3 = 0.9375;
    const double b4 = 1.21875;
    const double b5 = 1.53125;
    const double b6 = 1.84375;
    const double b7 = 2.078125;
    const double b8 = 2.3125;
    const double b9 = 2.734375;
    const double b10 = 3.15625;
    const double b11 = 4.203125;
    const double b12 = 4.6875;
    const double b13 = 6.1875;
    const double b14 = 7.76953125;
    const double b15 = 9.359375;
    const double b16 = 10.9375;
    const double b17 = 12.5625;

    double ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_Y1);
        double ch, cl, c0l, c1l;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0;
                    cl = 0.0;
                    c0l = 0.0;
                    c1l = 0.0;
                    p += 0*15;
                } else if (x < b1) {
                    ch = 0x1.0p-1;
                    cl = 0.0;
                    c0l = 0x1.8e6c89e55ffd9p-55;
                    c1l = 0.0;
                    p += 1*15;
                } else if (x < b2) {
                    ch = 0x1.4p-1;
                    cl = 0.0;
                    c0l = 0x1.4ef520d26aa2dp-54;
                    c1l = 0.0;
                    p += 2*15;
                } else if (x < b3) {
                    ch = 0x1.8p-1;
                    cl = 0.0;
                    c0l = 0x1.da45ff449a485p-58;
                    c1l = 0.0;
                    p += 3*15;
                } else {
                    ch = 0x1.ep-1;
                    cl = 0.0;
                    c0l = -0x1.b94e4503a6568p-57;
                    c1l = 0.0;
                    p += 4*15;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.38p+0;
                    cl = 0.0;
                    c0l = 0x1.7e034f3b454fap-56;
                    c1l = 0.0;
                    p += 5*15;
                } else if (x < b6) {
                    ch = 0x1.88p+0;
                    cl = 0.0;
                    c0l = -0x1.d4d7f06a836d6p-56;
                    c1l = 0.0;
                    p += 6*15;
                } else if (x < b7) {
                    ch = 0x1.d8p+0;
                    cl = 0.0;
                    c0l = 0x1.3a133555bd0f2p-58;
                    c1l = 0.0;
                    p += 7*15;
                } else {
                    ch = 0x1.193bed4dff243p+1;
                    cl = -0x1.bd1e50d219bfdp-55;
                    c0l = 0.0;
                    c1l = -0x1.48c9a75e5861fp-56;
                    p += 8*15;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.28p+1;
                    cl = 0.0;
                    c0l = 0x1.93755fb77dfdap-59;
                    c1l = 0.0;
                    p += 9*15;
                } else if (x < b10) {
                    ch = 0x1.5ep+1;
                    cl = 0.0;
                    c0l = -0x1.9422950a65efep-59;
                    c1l = 0.0;
                    p += 10*15;
                } else if (x < b11) {
                    ch = 0x1.d76d4affba175p+1;
                    cl = 0x1.3bac0714e4129p-58;
                    c0l = 0x1.010292e75474ep-58;
                    c1l = 0.0;
                    p += 11*15;
                } else if (x < b12) {
                    ch = 0x1.0dp+2;
                    cl = 0.0;
                    c0l = -0x1.7d5839241f5d1p-56;
                    c1l = 0.0;
                    p += 12*15;
                } else {
                    ch = 0x1.5b7fe4e87b02ep+2;
                    cl = 0x1.dfe7bac228e8cp-52;
                    c0l = 0.0;
                    c1l = -0x1.f1dae1892287ap-56;
                    p += 13*15;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.bc41890588553p+2;
                    cl = 0x1.7960b6b1c46acp-53;
                    c0l = -0x1.7ab9fc6d28237p-59;
                    c1l = 0.0;
                    p += 14*15;
                } else if (x < b15) {
                    ch = 0x1.13127ae6169b4p+3;
                    cl = 0x1.479cc068d9046p-52;
                    c0l = 0.0;
                    c1l = 0x1.465a98a39d5e1p-57;
                    p += 15*15;
                } else if (x < b16) {
                    ch = 0x1.43f2ee51e8c7ep+3;
                    cl = 0x1.8f4ba5d68e44p-51;
                    c0l = -0x1.754c4720f6a96p-57;
                    c1l = 0.0;
                    p += 16*15;
                } else {
                    ch = 0x1.77f9138d43206p+3;
                    cl = 0x1.0fc786ce0608p-55;
                    c0l = 0.0;
                    c1l = 0x1.ded19e09dda57p-58;
                    p += 17*15;
                }
            }
        }

        double x2 = x*x;
        // t + tl = (x < b0 ? x2 : x) - ch - cl; split the last Horner step
        double th = (x < b0 ? x2 : x) - ch;
        double t = th - cl;
        double tl = (th - t) - cl;
        double s = PE12(t, p[14], p[13], p[12], p[11], p[10], p[9], p[8], p[7], p[6], p[5], p[4], p[3], p[2]);
        double rh = BUILTIN_FMA_F64(t, s, p[1]);
        double d = p[1] - rh;
        double rl = BUILTIN_FMA_F64(t, s, d) + (p[1] - (d + rh));

        // p[0] + t*rh exactly, then the small terms with a single final rounding
        double ph = t * rh;
        double pl = BUILTIN_FMA_F64(t, rh, -ph);
        double a = p[0] + ph;
        double bb = a - p[0];
        double er = ((p[0] - (a - bb)) + (ph - bb)) + pl;
        double e = BUILTIN_FMA_F64(t, rl + c1l, er + BUILTIN_FMA_F64(tl, BUILTIN_FMA_F64(2.0 * t, p[2], p[1]), c0l));
        ret = a + e;

        if (x < b0) {
            // (2/pi)(j1(x) log(x) - 1/x) + x*(a + e) with a single final rounding
            double jm = x2 * PE6(x2, -0x1.522a43f65486ap-42, 0x1.27e4fb7789f5cp-34, -0x1.845c8a0ce5129p-27,
                                     0x1.6c16c16c16c17p-20, -0x1.c71c71c71c71cp-14, 0x1.5555555555555p-8, -0x1.0p-3);
            double jh = 1.0 + jm;
            double2 j = mul(0.5 * x, con(jh, (1.0 - jh) + jm));
            double q = MATH_FAST_RCP(x);
            double2 r = fadd(q, q * BUILTIN_FMA_F64(-x, q, 1.0));
            double2 v = mul(con(0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55), sub(mul(j, MATH_PRIVATE(epln)(x)), r));
            double xa = x * a;
            double sh = v.hi + xa;
            double sb = sh - v.hi;
            double sl = ((v.hi - (sh - sb)) + (xa - sb)) + (v.lo + BUILTIN_FMA_F64(x, e, BUILTIN_FMA_F64(x, a, -xa)));
            ret = sh + sl;

            // subnormal x: -(2/pi)/x, scaled so the reciprocal sees a normal input
            double xs = BUILTIN_ABS_F64(x) * 0x1.0p+64;
            double rs = MATH_FAST_RCP(xs);
            double qs = -0x1.45f306dc9c883p-1 * rs;
            qs = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-qs, xs, -0x1.45f306dc9c883p-1) + 0x1.6b01ec5417056p-55, rs, qs);
            qs = xs == 0.0 ? NINF_F64 : qs * 0x1.0p+64;
            ret = x < 0x1.0p-1022 ? qs : ret;
            ret = x < 0.0 ? QNAN_F64 : ret;
        }
    } else {
        double r = MATH_RCP(x);
        double r2 = r*r;
        double p = MATH_PRIVATE(bp1)(r2) * r;
        ret = 0x1.9884533d43651p-1 * MATH_FAST_SQRT(r) * MATH_PRIVATE(ba1)(r2) * MATH_PRIVATE(sinb)(x, 1, p);
        ret = x == PINF_F64 ? 0.0 : ret;
    }

    return ret;
}

