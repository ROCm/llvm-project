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
        double qt = PE13(y, -0x1.aed75feec7b9ap-23, 0x1.31854a0be3cd3p-20, -0x1.5037d6a97a8b7p-20,
                          -0x1.51d67f2cdbcfbp-16, 0x1.0c8ab2ac5112dp-13, -0x1.c364ce9b5e149p-13,
                          -0x1.317113a39f929p-10, 0x1.d919c501178a3p-8, -0x1.3b4af282da690p-7,
                          -0x1.59af103bf2cd0p-5, 0x1.5512320b432ccp-3, -0x1.5815e8fa28886p-5,
                          -0x1.4fcf4026afa24p-1, 0x1.2788cfc6fb61cp-1);

        double2 den = fadd(mul(d, y*qt), d);
        ret = MATH_DIV(n.hi, den.hi);
        ret = x == 0.0 ? BUILTIN_COPYSIGN_F64(PINF_F64, x) : ret;
        ret = x < 0.0 && BUILTIN_TRUNC_F64(x) == x ? QNAN_F64 : ret;
    } else {
        double xr = MATH_FAST_RCP(ax);
        double pt = PE6(xr, -0x1.2b04c5ea74bbfp-11, 0x1.14869344f1d9bp-14, 0x1.9b3457156ffefp-11,
                          -0x1.e1427e86ee097p-13, -0x1.5f7266f67c4e0p-9, 0x1.c71c71c0f96adp-9,
                          0x1.5555555555a28p-4);

        double2 e = sub(mul(MATH_PRIVATE(epln)(ax), ax - 0.5), ax);

        if (x > 0.0) {
            const double sqrt2pi = 0x1.40d931ff62706p+1;
            double m = MATH_PRIVATE(expep)(e);
            double g = MATH_MAD(sqrt2pi*m, xr*pt, sqrt2pi*m);
            ret = x > 0x1.573fae561f646p+7 ? PINF_F64 : g;
        } else {
            const double2 lnsqrtpiby2 = con(0x1.ce6bb25aa1316p-3, -0x1.dcd49c8e5aff6p-58);
            double s = -x * MATH_MANGLE(sinpi)(x);
            double asp = BUILTIN_ABS_F64(s) * MATH_MAD(xr, pt, 1.0);
            double mag = MATH_PRIVATE(expep)(sub(sub(lnsqrtpiby2, e), MATH_PRIVATE(epln)(asp)));
            ret = BUILTIN_COPYSIGN_F64(mag, s);
            ret = BUILTIN_TRUNC_F64(x) == x || BUILTIN_ISNAN_F64(x) ? QNAN_F64 : ret;
        }
    }

    return ret;
}
