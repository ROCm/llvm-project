/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(atan2ep)(float2 y, float2 x)
{
    const float piby2_h = 0x1.921fb6p+0f;
    const float piby2_l = -0x1.777a5cp-25f;
    const float pi_h = 0x1.921fb6p+1f;
    const float pi_l = -0x1.777a5cp-24f;

    float ax = BUILTIN_ABS_F32(x.hi);
    float ay = BUILTIN_ABS_F32(y.hi);
    float axl = AS_FLOAT(AS_UINT(x.lo) ^ (AS_UINT(x.hi) & SIGNBIT_SP32));
    float ayl = AS_FLOAT(AS_UINT(y.lo) ^ (AS_UINT(y.hi) & SIGNBIT_SP32));
    bool sw = ay > ax;
    float u = sw ? ay : ax;
    float ul = sw ? ayl : axl;
    float v = sw ? ax : ay;
    float vl = sw ? axl : ayl;

    // Keep u, its reciprocal, and the division residual normal: scale small
    // u up into [0.5,1), and large u down only as far as 2^124
    bool ok = BUILTIN_CLASS_F32(u, CLASS_PNOR|CLASS_PSUB);
    int eu = BUILTIN_FREXP_EXP_F32(u);
    int sc = eu < 0 ? -eu : (eu > 124 ? 124 - eu : 0);
    sc = ok ? sc : 0;

    // Scale a tiny v up so that no intermediate goes denormal
    int kv = (ok & (BUILTIN_FREXP_EXP_F32(v) - eu < -100)) ? 64 : 0;
    u = BUILTIN_FLDEXP_F32(u, sc);
    ul = BUILTIN_FLDEXP_F32(ul, sc);
    v = BUILTIN_FLDEXP_F32(v, sc + kv);
    vl = BUILTIN_FLDEXP_F32(vl, sc + kv);

    // r + d = (v + vl) / (u + ul)
    float ru = BUILTIN_AMDGPU_RCP_F32(u);
    float r = v * ru;
    float d = BUILTIN_FMA_F32(-r, ul, BUILTIN_FMA_F32(-r, u, v) + vl) * ru;

    // a + c = atan(r + d)
    float t = r * r;
    float w = t * PE7(t, 0x1.5a54bp-9f, -0x1.f4b218p-7f, 0x1.53f67ep-5f, -0x1.2fa9aep-4f, 0x1.b26364p-4f,
                         -0x1.22c1ccp-3f, 0x1.99717ep-3f, -0x1.5554c4p-2f);
    float p = r * w;
    float a = r + p;
    float c = ((r - a) + p) + BUILTIN_FMA_F32(r, w, -p);
    c = BUILTIN_FMA_F32(d, BUILTIN_AMDGPU_RCP_F32(BUILTIN_FMA_F32(r, r, 1.0f)), c);

    float rs = BUILTIN_FLDEXP_F32(a + c, -kv);
    a = BUILTIN_FLDEXP_F32(a, -kv);
    c = BUILTIN_FLDEXP_F32(c, -kv);

    float s = piby2_h - a;
    float sl = ((piby2_h - s) - a) + (piby2_l - c);
    a = sw ? s : a;
    c = sw ? sl : c;

    bool xneg = AS_INT(x.hi) < 0;
    s = pi_h - a;
    sl = ((pi_h - s) - a) + (pi_l - c);
    a = xneg ? s : a;
    c = xneg ? sl : c;

    float ret = ok ? ((sw | xneg) ? a + c : rs) : a;

    float tt = xneg ? pi_h : 0.0f;
    ret = y.hi == 0.0f ? tt : ret;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        tt = xneg ? 0x1.2d97c8p+1f : 0x1.921fb6p-1f;
        ret = (BUILTIN_ISINF_F32(x.hi) & BUILTIN_ISINF_F32(y.hi)) ? tt : ret;

        // x or y is NaN
        ret = BUILTIN_ISUNORDERED_F32(x.hi, y.hi) ? QNAN_F32 : ret;
    }

    return BUILTIN_COPYSIGN_F32(ret, y.hi);
}
