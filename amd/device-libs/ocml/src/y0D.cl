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
extern CONSTATTR double MATH_PRIVATE(bp0)(double);
extern CONSTATTR double MATH_PRIVATE(ba0)(double);

CONSTATTR double
MATH_MANGLE(y0)(double x)
{
    const double b0  = 0.3125;
    const double b1  = 0.4375;
    const double b2  = 0.5625;
    const double b3  = 0.6875;
    const double b4  = 0.8125;
    const double b5  = 1.0;
    const double b6  = 1.25;
    const double b7  = 1.625;
    const double b8  = 2.0;
    const double b9  = 2.53125;
    const double b10 = 3.0;
    const double b11 = 3.484375;
    const double b12 = 4.703125;
    const double b13 = 6.265625;
    const double b14 = 7.84375;
    const double b15 = 9.421875;
    const double b16 = 10.984375;
    const double b17 = 12.546875;
    double ret;
    double xin = x;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_Y0);
        double ch, cl, c0l, c1l;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0;
                    cl = 0.0;
                    c0l = 0.0;
                    c1l = 0.0;
                } else if (x < b1) {
                    ch = 0x1.4p-2;
                    cl = 0.0;
                    c0l = -0x1.abd1932f8f358p-55;
                    c1l = 0.0;
                    p += 1*15;
                } else if (x < b2) {
                    ch = 0x1.cp-2;
                    cl = 0.0;
                    c0l = 0x1.75c127ffea936p-58;
                    c1l = 0.0;
                    p += 2*15;
                } else if (x < b3) {
                    ch = 0x1.2p-1;
                    cl = 0.0;
                    c0l = -0x1.37eb787151c69p-56;
                    c1l = 0.0;
                    p += 3*15;
                } else {
                    ch = 0x1.6p-1;
                    cl = 0.0;
                    c0l = -0x1.c2d0a8f30d454p-61;
                    c1l = 0.0;
                    p += 4*15;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.c982eb8d417eap-1;
                    cl = 0x1.ea9d270347f83p-56;
                    c0l = 0.0;
                    c1l = -0x1.1cfcab20b2389p-56;
                    p += 5*15;
                } else if (x < b6) {
                    ch = 0x1.p+0;
                    cl = 0.0;
                    c0l = 0x1.1a878a50773f6p-58;
                    c1l = 0.0;
                    p += 6*15;
                } else if (x < b7) {
                    ch = 0x1.4p+0;
                    cl = 0.0;
                    c0l = 0x1.ce9ea2ac779eap-57;
                    c1l = 0.0;
                    p += 7*15;
                } else {
                    ch = 0x1.ap+0;
                    cl = 0.0;
                    c0l = 0x1.ccd4c3d09b3c9p-56;
                    c1l = 0.0;
                    p += 8*15;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.193bed4dff243p+1;
                    cl = -0x1.bd1e50d219bfdp-55;
                    c0l = -0x1.5596507017014p-56;
                    c1l = 0.0;
                    p += 9*15;
                } else if (x < b10) {
                    ch = 0x1.44p+1;
                    cl = 0.0;
                    c0l = -0x1.dd1cff5dbf5d8p-57;
                    c1l = 0.0;
                    p += 10*15;
                } else if (x < b11) {
                    ch = 0x1.8p+1;
                    cl = 0.0;
                    c0l = -0x1.fadd5b45832abp-57;
                    c1l = 0.0;
                    p += 11*15;
                } else if (x < b12) {
                    ch = 0x1.fa9534d98569cp+1;
                    cl = -0x1.f06ae7804384ep-54;
                    c0l = 0.0;
                    c1l = -0x1.41b272f31b373p-57;
                    p += 12*15;
                } else {
                    ch = 0x1.5b7fe4e87b02ep+2;
                    cl = 0x1.dfe7bac228e8cp-52;
                    c0l = 0x1.9b0f5a8825172p-56;
                    c1l = 0.0;
                    p += 13*15;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.c581dc4e72103p+2;
                    cl = -0x1.9774a495f56cfp-54;
                    c0l = 0.0;
                    c1l = -0x1.8c11f65487191p-56;
                    p += 14*15;
                } else if (x < b15) {
                    ch = 0x1.13127ae6169b4p+3;
                    cl = 0x1.479cc068d9046p-52;
                    c0l = 0x1.d5bce8f18636dp-59;
                    c1l = 0.0;
                    p += 15*15;
                } else if (x < b16) {
                    ch = 0x1.471d735a47d58p+3;
                    cl = -0x1.cb49ff791c495p-51;
                    c0l = 0.0;
                    c1l = 0x1.5cfa6318f5a77p-57;
                    p += 16*15;
                } else {
                    ch = 0x1.77f9138d43206p+3;
                    cl = 0x1.0fc786ce0608p-55;
                    c0l = 0x1.b233f863a532dp-58;
                    c1l = 0.0;
                    p += 17*15;
                }
            }
        }

        // ret + retl = (2/pi) j0(x) log(x) for small x
        double retl = 0.0;
        ret = 0.0;
        if (x < b0) {
            double x2 = x * x;
            double jm = x2 * PE5(x2, 0x1.02e85c0898b71p-31, -0x1.23456789abcdfp-24, 0x1.c71c71c71c71cp-18,
                                     -0x1.c71c71c71c71cp-12, 0x1.0p-6, -0x1.0p-2);
            double jh = 1.0 + jm;
            double2 v = mul(mul(con(0x1.45f306dc9c883p-1, -0x1.6b01ec5417056p-55), MATH_PRIVATE(epln)(x)),
                            con(jh, (1.0 - jh) + jm));
            ret = v.hi;
            retl = v.lo;
            x = x2;
        }

        // t + tl = x - ch - cl; split the last Horner step
        double th = x - ch;
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
        double sh = ret + a;
        double sb = sh - ret;
        ret = sh + ((((ret - (sh - sb)) + (a - sb)) + retl) + e);

        if (!FINITE_ONLY_OPT()) {
            ret = xin < 0.0 ? QNAN_F64 : ret;
            ret = xin == 0.0 ? NINF_F64 : ret;
        }

    } else {
        double r = MATH_RCP(x);
        double r2 = r*r;
        double p = MATH_PRIVATE(bp0)(r2) * r;
        ret = 0x1.9884533d43651p-1 * MATH_FAST_SQRT(r) * MATH_PRIVATE(ba0)(r2) * MATH_PRIVATE(sinb)(x, 0, p);
        ret = x == PINF_F64 ? 0.0 : ret;
    }

    return ret;
}

