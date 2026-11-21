/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rsqrt)(double x)
{
    double y0 = BUILTIN_AMDGPU_RSQRT_F64(x);

    // e = 1 - x*y0^2 with x*y0 exact
    double t = x * y0;
    double e = MATH_MAD(-MATH_MAD(x, y0, -t), y0, MATH_MAD(-t, y0, 1.0));
    double y = MATH_MAD(y0*e, MATH_MAD(e, 0.375, 0.5), y0);

    return BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_NZER|CLASS_PINF) ? y0 : y;
}

