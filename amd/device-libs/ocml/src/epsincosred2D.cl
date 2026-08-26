/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

#include "trigredD.h"

CONSTATTR double4
MATH_PRIVATE(epsincosred2)(double x, double y)
{
    double2 a = fadd(x, y);
    double2 a2 = sqr(a);
    double2 a3 = mul(a2, a);
    double2 a4 = sqr(a2);
    double t = a2.hi;

    const double2 C0 = (double2)(0x1.5555555555555p-59, 0x1.5555555555555p-5);
    double pc = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    0x1.ae7f3e733b81fp-45, -0x1.93974a8c07c9dp-37), 0x1.1eed8eff8d898p-29),
                    -0x1.27e4fb7789f5cp-22), 0x1.a01a01a01a01ap-16), -0x1.6c16c16c16c17p-10);
    double2 c = add(fsub(1.0, ldx(a2, -1)), mul(a4, add(C0, mul(a2, pc))));

    const double2 S0 = (double2)(-0x1.5555555555555p-57, -0x1.5555555555555p-3);
    double ps = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    0x1.952c77030ad4ap-49, -0x1.ae7f3e733b81fp-41), 0x1.6124613a86d09p-33),
                    -0x1.ae64567f544e4p-26), 0x1.71de3a556c734p-19), -0x1.a01a01a01a01ap-13),
                    0x1.1111111111111p-7);
    double2 s = add(a, mul(a3, add(S0, mul(a2, ps))));

    return (double4)(c, s);
}
