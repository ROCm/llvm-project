/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(hypot)

CONSTATTR half
MATH_MANGLE(hypot)(half x, half y)
{
    float fx = (float)x;
    float fy = (float)y;
    float d2 = BUILTIN_FMA_F32(fx, fx, fy*fy);
    return (half)BUILTIN_AMDGPU_SQRT_F32(d2);
}

