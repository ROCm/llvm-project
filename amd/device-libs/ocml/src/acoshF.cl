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
MATH_MANGLE(acosh)(float x)
{
    bool b = x >= 0x1.0p+64f;
    float s = b ? 0x1.0p-64f : 1.0f;
    float sx = x * s;
    float s2 = s * s;

    // h = sx^2 - s^2 = hh + hl, where sx^2 >= s^2
    float pp = sx * sx;
    float pl = BUILTIN_FMA_F32(sx, sx, -pp);
    float hh = pp - s2;
    float hl = ((pp - hh) - s2) + pl;

    // a = sx + sqrt(h), where sx >= sqrt(h)
    float rh = MATH_FAST_SQRT(hh);
    float rl = (BUILTIN_FMA_F32(-rh, rh, hh) + hl) * (0.5f * MATH_FAST_RCP(rh));
    rl = hh == 0.0f ? 0.0f : rl;
    float ah = sx + rh;
    float2 a = con(ah, ((sx - ah) + rh) + rl);
    float z = MATH_PRIVATE(lnep)(a, b ? 64 : 0);

    if (!FINITE_ONLY_OPT()) {
        z = x == PINF_F32 ? x : z;
        z = x < 1.0f ? QNAN_F32 : z;
    }

    return z;
}

