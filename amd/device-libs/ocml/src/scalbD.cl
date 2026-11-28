/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(scalb)(double x, double y)
{
    double t = BUILTIN_MIN_F64(BUILTIN_MAX_F64(y, -0x1.0p+20), 0x1.0p+20);
    double n = BUILTIN_RINT_F64(t);
    double f = t - n;

    // q = 2^f - 1, |f| <= 0.5
    double p = PE10(f, 0x1.bd1db697ac267p-36, 0x1.e9e590a64d9c9p-32, 0x1.e4d48e0226f4bp-28, 0x1.b524f03a01d19p-24,
                      0x1.62c020c5f7987p-20, 0x1.ffcbfc6b84327p-17, 0x1.430912f8c8e0dp-13,
                      0x1.5d87fe78a46bap-10, 0x1.3b2ab6fba4d3ap-7, 0x1.c6b08d704a0c4p-5, 0x1.ebfbdff82c58fp-3);
    double q = MATH_MAD(f, 0x1.62e42fefa39efp-1, MATH_MAD(f, 0x1.abc9e3b39803fp-56, f * f * p));

    // Scale the normalized mantissa so that only the final ldexp can round again
    int e;
    double m = BUILTIN_FREXP_F64(x, &e);
    double ret = BUILTIN_FLDEXP_F64(MATH_MAD(m, q, m), e + (int)n);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F64(x) ? x : ret;
        ret = BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;
        ret = ((x == 0.0) & (y == PINF_F64)) ? QNAN_F64 : ret;
        ret = (BUILTIN_ISINF_F64(x) & (y == NINF_F64)) ? QNAN_F64 : ret;
    }

    return ret;
}

