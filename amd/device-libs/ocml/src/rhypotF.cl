/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(rhypot)(float x, float y)
{
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(a), AS_UINT(b)));
    int e = BUILTIN_FREXP_EXP_F32(t);
    a = BUILTIN_FLDEXP_F32(a, -e);
    b = BUILTIN_FLDEXP_F32(b, -e);
#if defined EXTRA_ACCURACY
    float b2 = b * b;
    float d2 = BUILTIN_FMA_F32(a, a, b2);
    float z = BUILTIN_AMDGPU_RSQRT_F32(d2);

    // One Newton step, with the residual 1 - (d2 + b2l)*z^2 including the
    // error of d2*z and the rounding error of b*b
    float p = d2 * z;
    float u = BUILTIN_FMA_F32(-p, z, 1.0f);
    u = BUILTIN_FMA_F32(-BUILTIN_FMA_F32(d2, z, -p), z, u);
    u = BUILTIN_FMA_F32(-BUILTIN_FMA_F32(b, b, -b2), z*z, u);
    float zr = BUILTIN_FMA_F32(0.5f*z, u, z);
    z = t == 0.0f ? z : zr;
    float ret = BUILTIN_FLDEXP_F32(z, -e);
#else
    float ret = BUILTIN_FLDEXP_F32(BUILTIN_AMDGPU_RSQRT_F32(MATH_MAD(a, a, b*b)), -e);
#endif

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISINF_F32(x) |
               BUILTIN_ISINF_F32(y)) ?
              0.0f : ret;
    }

    return ret;
}

