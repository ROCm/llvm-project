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
MATH_MANGLE(asinh)(float x)
{
    float y = BUILTIN_ABS_F32(x);
    bool b = y >= 0x1.0p+64f;
    float s = b ? 0x1.0p-64f : 1.0f;
    float sy = y * s;
    float s2 = s * s;

    // h = sy^2 + s^2 = hh + hl
    float pp = sy * sy;
    float pl = BUILTIN_FMA_F32(sy, sy, -pp);
    float hh = pp + s2;
    float hb = hh - pp;
    float hl = ((pp - (hh - hb)) + (s2 - hb)) + pl;

    // a = sy + sqrt(h), where sqrt(h) >= sy
    float rh = MATH_FAST_SQRT(hh);
    float rl = (BUILTIN_FMA_F32(-rh, rh, hh) + hl) * (0.5f * MATH_FAST_RCP(rh));
    float ah = rh + sy;
    float2 a = con(ah, ((rh - ah) + sy) + rl);
    float z = MATH_PRIVATE(lnep)(a, b ? 64 : 0);

    z = y < 0x1.0p-12f ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = y == PINF_F32 ? y : z;
    }

    return BUILTIN_COPYSIGN_F32(z, x);
}

