
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(atan2)

CONSTATTR half
MATH_MANGLE(atan2)(half y, half x)
{
    const half pi = 0x1.921fb6p+1h;
    const half piby4 = 0x1.921fb6p-1h;
    const half threepiby4 = 0x1.2d97c8p+1h;

    // Evaluate in float so that only the final conversion rounds to half
    float fax = BUILTIN_ABS_F32((float)x);
    float fay = BUILTIN_ABS_F32((float)y);
    float v = BUILTIN_MIN_F32(fax, fay);
    float u = BUILTIN_MAX_F32(fax, fay);

    float vbyu = v * BUILTIN_AMDGPU_RCP_F32(u);
    float vv = vbyu * vbyu;
    float p = BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, 0x1.96fbc8p-6f, -0x1.816ce6p-4f), 0x1.7e9874p-3f), -0x1.541a14p-2f);
    float fa = BUILTIN_FMA_F32(vv, vbyu*p, vbyu);

    float ft = 0x1.921fb6p+0f - fa;
    fa = fay > fax ? ft : fa;
    ft = 0x1.921fb6p+1f - fa;
    fa = x < 0.0h ? ft : fa;

    half a = (half)fa;
    half t = AS_SHORT(x) < 0 ? pi : 0.0h;
    a = y == 0.0h ? t : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        t = x < 0.0h ? threepiby4 : piby4;
        a = (BUILTIN_ISINF_F16(x) & BUILTIN_ISINF_F16(y)) ? t : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F16(x, y) ? QNAN_F16 : a;
    }

    return BUILTIN_COPYSIGN_F16(a, y);
}

