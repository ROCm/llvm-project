/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if !defined EXTRA_ACCURACY
#define EXTRA_ACCURACY
#endif

CONSTATTR float
MATH_MANGLE(cbrt)(float x)
{
    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

#if defined EXTRA_ACCURACY
    // Scale small inputs so that the residual ax - z^3 stays normal
    float ax = BUILTIN_ABS_F32(x);
    bool small = ax < 0x1p-92f;
    ax = small ? BUILTIN_FLDEXP_F32(ax, 96) : ax;

    float z = BUILTIN_AMDGPU_EXP2_F32(0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32(ax));

    // One Newton step with z^3 = q + ql exact
    float p = z * z;
    float pl = BUILTIN_FMA_F32(z, z, -p);
    float q = p * z;
    float ql = BUILTIN_FMA_F32(pl, z, BUILTIN_FMA_F32(p, z, -q));
    float d = (ax - q) - ql;
    z = BUILTIN_FMA_F32(d, 0x1.555556p-2f * MATH_FAST_RCP(p), z);

    z = small ? BUILTIN_FLDEXP_F32(z, -32) : z;
#else
    float ax = BUILTIN_ABS_F32(x);
    bool denorm_or_zero = ax < 0x1p-126f;

    if (!DAZ_OPT()) {
        ax = denorm_or_zero ?
             BUILTIN_FLDEXP_F32(ax, 24) : ax;
    }

    float z = BUILTIN_AMDGPU_EXP2_F32(0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32(ax));
    z = MATH_MAD(MATH_MAD(MATH_FAST_RCP(z*z), -ax, z), -0x1.555556p-2f, z);

    if (!DAZ_OPT()) {
        z = denorm_or_zero ?
            BUILTIN_FLDEXP_F32(z, -8) : z;
    }
#endif

    // Is normal or subnormal.
    z = x == 0.0f || BUILTIN_ISINF_F32(x) ? x : z;
    return BUILTIN_COPYSIGN_F32(z, x);
}

