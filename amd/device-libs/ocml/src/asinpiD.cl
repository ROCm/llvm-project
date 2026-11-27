/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double
MATH_MANGLE(asinpi)(double x)
{
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    double y = BUILTIN_ABS_F64(x);
    bool transform = y >= 0.5;

    double rt = MATH_MAD(y, -0.5, 0.5);
    double y2 = y * y;
    double r = transform ? rt : y2;

    double u = PE12(r, 0x1.4ec5bf9642e32p-7, -0x1.586f90fb93cfep-8, 0x1.9b7d273c35116p-8, 0x1.0d15d60b04579p-9,
                       0x1.fc8cc693d7332p-9, 0x1.2195fc3a46e55p-8, 0x1.6a250ac354707p-8,
                       0x1.d2b0f0dd5e719p-8, 0x1.3ce5354d7daa4p-7, 0x1.d1a452b2c77c8p-7,
                       0x1.8723a1d5e63b4p-6, 0x1.b2995e7b7b29ep-5, -0x1.176f5c1ab9f63p-56);

    const double piinv = 0x1.45f306dc9c883p-2;
    double v = MATH_MAD(y, piinv, y*u);
    if (transform) {
        double2 s = ldx(root2(r), 1);
        double2 ve = fsub(0.5, fadd(mul(piinv, s), mul(s, u)));
        v = ve.hi;
        v = y == 1.0 ? 0.5 : v;
    }

    return BUILTIN_COPYSIGN_F64(v, x);
}

