/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(len4)(half x, half y, half z, half w)
{
    float fx = (float)x;
    float fy = (float)y;
    float fz = (float)z;
    float fw = (float)w;
    float d2 = BUILTIN_FMA_F32(fx, fx, BUILTIN_FMA_F32(fy, fy, BUILTIN_FMA_F32(fz, fz, fw*fw)));
    return (half)BUILTIN_AMDGPU_SQRT_F32(d2);
}

