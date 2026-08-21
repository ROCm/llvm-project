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
    float ret = MATH_MANGLE(ldexp)(x, (int)n);
    ret = f == 0.0f ? ret : ret * MATH_MANGLE(exp2)(f);

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : ret;
        ret = ((x == 0.0f) & (y == PINF_F32)) ? QNAN_F32 : ret;
        ret = (BUILTIN_ISINF_F32(x) & (y == NINF_F32)) ? QNAN_F32 : ret;
    }

    return ret;
}

