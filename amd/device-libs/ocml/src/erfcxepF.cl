/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

// erfcx(x) as hi + lo for x >= 0
CONSTATTR float2
MATH_PRIVATE(erfcxep)(float x)
{
    float n = x - 2.0f;
    float d = x + 2.0f;
    float r = MATH_FAST_RCP(d);
    float q = n * r;
    float e = BUILTIN_FMA_F32(-q, x, BUILTIN_FMA_F32(q + 1.0f, -2.0f, x));
    q = BUILTIN_FMA_F32(r, e, q);
    
    float g = PE10(q, 0x1.3d95d4p-14f, 0x1.727f88p-14f, -0x1.fef738p-12f,
                    -0x1.e779e6p-11f, 0x1.1e0d72p-9f, 0x1.abc1a8p-8f,
                    -0x1.dd06d2p-7f, -0x1.44e6f4p-5f, 0x1.a133acp-3f,
                    -0x1.7a9db2p-2f, 0x1.1ba038p-2f);
    float p = BUILTIN_FMA_F32(q, g, g);
    float tx = x + x;
    d = 1.0f + tx;
    r = MATH_FAST_RCP(d);
    q = BUILTIN_FMA_F32(p, r, r);
    // residual 1 + p - q(1 + 2x); for x < 0.75, q >= 0.5 so 1 - q is exact
    bool sm = x < 0.75f;
    e = ((sm ? 1.0f : p) - q) + BUILTIN_FMA_F32(-q, tx, sm ? p : 1.0f);
    return con(q, r * e);
}
