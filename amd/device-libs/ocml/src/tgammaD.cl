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
extern CONSTATTR double MATH_PRIVATE(expep)(double2);

CONSTATTR double
MATH_MANGLE(tgamma)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 16.0) {
        double2 n, d;
        double y = x;
        if (x > 0.0) {
            n = con(1.0, 0.0);
            while (y > 2.5) {
                n = omul(n, y - 1.0);
                y = y - 1.0;
                n = omul(n, y - 1.0);
                y = y - 1.0;
            }
            if (y > 1.5) {
                n = omul(n, y - 1.0);
                y = y - 1.0;
            }
            if (x >= 0.5)
                y = y - 1.0;
            d = con(x < 0.5 ? x : 1.0, 0.0);
        } else {
            d = con(x, 0.0);
            while (y < -1.5) {
                d = omul(d, y + 1.0);
                y = y + 1.0;
                d = omul(d, y + 1.0);
                y = y + 1.0;
            }
            if (y < -0.5) {
                d = omul(d, y + 1.0);
                y = y + 1.0;
            }
            n = con(1.0, 0.0);
        }
        // g + gl = 1/Gamma(1 + y)
        double q = PE15(y, 0x1.66c534675cf87p-28, 0x1.4e292c4032facp-28, -0x1.b9ef237109715p-23, 0x1.304acb6af0cp-20,
                        -0x1.4fa968a0f419fp-20, -0x1.51cf17a359656p-16, 0x1.0c8a7749ee0bdp-13,
                        -0x1.c364fc196accdp-13, -0x1.317112c8cae4ap-10, 0x1.d919c5268d97bp-8,
                        -0x1.3b4af2848ea4dp-7, -0x1.59af103c323b8p-5, 0x1.5512320b44013p-3,
                        -0x1.5815e8fa270cdp-5, -0x1.4fcf4026afa2ep-1, 0x1.2788cfc6fb619p-1);
        double g = BUILTIN_FMA_F64(y, q, 0x1p+0);
        double gl = BUILTIN_FMA_F64(y, q, 0x1p+0 - g) + -0x1.53bfe0b67e704p-60;
        double2 den = mul(d, con(g, gl));

        // (n.hi + n.lo) / (den.hi + den.lo); scale a tiny den.hi so the reciprocal
        // sees a normal input
        double sc = ax < 0x1.0p-512 ? 0x1.0p+512 : 1.0;
        double r = MATH_FAST_RCP(den.hi * sc) * sc;
        double qq = n.hi * r;
        ret = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-qq, den.lo, BUILTIN_FMA_F64(-qq, den.hi, n.hi) + n.lo), r, qq);
        ret = BUILTIN_ISINF_F64(qq) ? qq : ret;
        ret = x == 0.0 ? BUILTIN_COPYSIGN_F64(PINF_F64, x) : ret;
        ret = x < 0.0 && BUILTIN_TRUNC_F64(x) == x ? QNAN_F64 : ret;
    } else {
        // e = lgamma(ax) = (ax - 0.5)*ln(ax) - ax + ln(sqrt(2pi)) + c(1/ax)
        double xr = MATH_FAST_RCP(ax);
        double c = xr * PE6(xr*xr, 0x1.a41a41a41a41ap-8, -0x1.f6ab0d9993c7dp-10, 0x1.b951e2b18ff23p-11,
                            -0x1.3813813813814p-11, 0x1.a01a01a01a01ap-11, -0x1.6c16c16c16c17p-9,
                            0x1.5555555555555p-4);
        double2 e = fadd(fadd(sub(mul(MATH_PRIVATE(epln)(ax), ax - 0.5), ax),
                              con(0x1.d67f1c864beb5p-1, -0x1.65b5a1b7ff5dfp-55)), c);

        if (x > 0.0) {
            ret = MATH_PRIVATE(expep)(e);
            ret = x > 0x1.573fae561f646p+7 ? PINF_F64 : ret;
        } else {
            // |Gamma(x)| = pi / (|x*sinpi(x)| * Gamma(ax)), computed 2^512 times larger
            // so that exp of the high part stays normal for a normal result
            double sp = MATH_MANGLE(sinpi)(x);
            double sh = ax * BUILTIN_ABS_F64(sp);
            double sl = BUILTIN_FMA_F64(ax, BUILTIN_ABS_F64(sp), -sh);
            double2 ls = add(MATH_PRIVATE(epln)(sh), sl * MATH_FAST_RCP(sh));
            double2 l = sub(sub(con(0x1.64093cf432191p+8, -0x1.af5db206c71b3p-49), e), ls);
            double mag = BUILTIN_FLDEXP_F64(MATH_PRIVATE(expep)(l), -512);
            ret = BUILTIN_COPYSIGN_F64(mag, sp);
            ret = BUILTIN_TRUNC_F64(x) == x || BUILTIN_ISNAN_F64(x) ? QNAN_F64 : ret;
        }
    }

    return ret;
}
