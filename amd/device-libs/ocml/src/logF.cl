/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);

CONSTATTR float
MATH_MANGLE(log)(float x)
{
    float z = MATH_PRIVATE(lnep)(con(x, 0.0f), 0);

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_ISINF_F32(x) ? x : z;
        z = x < 0.0f ? QNAN_F32 : z;
        z = x == 0.0f ? NINF_F32 : z;
    }

    return z;
}
