
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(atan2pi)


CONSTATTR half
MATH_MANGLE(atan2pi)(half y, half x)
{
    // Evaluate in float so that only the final conversion rounds to half
    float fax = BUILTIN_ABS_F32((float)x);
    float fay = BUILTIN_ABS_F32((float)y);
    float v = BUILTIN_MIN_F32(fax, fay);
    float u = BUILTIN_MAX_F32(fax, fay);

    float vbyu = v * BUILTIN_AMDGPU_RCP_F32(u);
    float vv = vbyu * vbyu;
    float fa = vbyu * BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, 0x1.f1ca04p-8f, -0x1.df6742p-6f), 0x1.e2f692p-5f), -0x1.b07856p-4f), 0x1.45f086p-2f);

    float ft = 0.5f - fa;
    fa = fay > fax ? ft : fa;
    ft = 1.0f - fa;
    fa = x < 0.0h ? ft : fa;

    half a = (half)fa;
    half at = AS_SHORT(x) < 0 ? 1.0h : 0.0h;
    a = y == 0.0h ? at : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        at = x < 0.0h ? 0.75h : 0.25h;
        a = (BUILTIN_ISINF_F16(x) & BUILTIN_ISINF_F16(y)) ?
            at : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F16(x, y) ? QNAN_F16 : a;
    }

    return BUILTIN_COPYSIGN_F16(a, y);
}
