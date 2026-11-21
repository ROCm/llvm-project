/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(scalb)(float x, float y)
{
    float t = BUILTIN_CLAMP_F32(y, -0x1.0p+20f, 0x1.0p+20f);
    float n = BUILTIN_RINT_F32(t);
    float f = t - n;

    // q = 2^f - 1, |f| <= 0.5
    float p = PE4(f, 0x1.426dap-13f, 0x1.5eec02p-10f, 0x1.3b2c62p-7f, 0x1.c6afb2p-5f, 0x1.ebfbdep-3f);
    float q = BUILTIN_FMA_F32(f, 0x1.62e430p-1f, BUILTIN_FMA_F32(f, -0x1.05c610p-29f, f * f * p));

    // Scale the normalized mantissa so that only the final ldexp can round again
    int e;
    float m = BUILTIN_FREXP_F32(x, &e);
    float ret = BUILTIN_FLDEXP_F32(BUILTIN_FMA_F32(m, q, m), e + (int)n);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F32(x) ? x : ret;
        ret = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : ret;
        ret = ((x == 0.0f) & (y == PINF_F32)) ? QNAN_F32 : ret;
        ret = (BUILTIN_ISINF_F32(x) & (y == NINF_F32)) ? QNAN_F32 : ret;
    }

    return ret;
}

