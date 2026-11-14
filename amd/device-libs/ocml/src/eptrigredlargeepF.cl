/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

// 64-bit window of (M*2^(ve-23))*(2/pi): 2 integer lsbs in [63,62], fraction below.
static ulong
win64(uint M, int ve)
{
    if (M == 0)
        return 0UL;

    ulong a;
    a = (ulong)M * 0xfe5163abU;      uint p0 = (uint)a; a >>= 32;
    a = (ulong)M * 0x3c439041U + a;  uint p1 = (uint)a; a >>= 32;
    a = (ulong)M * 0xdb629599U + a;  uint p2 = (uint)a; a >>= 32;
    a = (ulong)M * 0xf534ddc0U + a;  uint p3 = (uint)a; a >>= 32;
    a = (ulong)M * 0xfc2757d1U + a;  uint p4 = (uint)a; a >>= 32;
    a = (ulong)M * 0x4e441529U + a;  uint p5 = (uint)a; a >>= 32;
    a = (ulong)M * 0xa2f9836eU + a;  uint p6 = (uint)a; uint p7 = (uint)(a >> 32);

    int sh = 185 - ve;
    if (sh >= 256)
        return 0UL;

    int c;
    c = sh >= 128;
    p0 = c ? p4 : p0; p1 = c ? p5 : p1; p2 = c ? p6 : p2; p3 = c ? p7 : p3;
    p4 = c ? 0U : p4; p5 = c ? 0U : p5;
    sh -= c ? 128 : 0;

    c = sh >= 64;
    p0 = c ? p2 : p0; p1 = c ? p3 : p1; p2 = c ? p4 : p2; p3 = c ? p5 : p3;
    sh -= c ? 64 : 0;

    c = sh >= 32;
    p0 = c ? p1 : p0; p1 = c ? p2 : p1; p2 = c ? p3 : p2;
    sh -= c ? 32 : 0;

    uint lo32 = BUILTIN_FSHR_B32(p1, p0, (uint)sh);
    uint hi32 = BUILTIN_FSHR_B32(p2, p1, (uint)sh);
    return ((ulong)hi32 << 32) | (ulong)lo32;
}

CONSTATTR struct epredret
MATH_PRIVATE(eptrigredlargeep)(float2 x)
{
    // Each part gets its own window; they add at the units place.
    int eh;
    float mh = BUILTIN_FREXP_F32(x.hi, &eh);
    --eh;
    uint Mh = (uint)BUILTIN_FLDEXP_F32(mh, 24);
    int el;
    float ml = BUILTIN_FREXP_F32(x.lo, &el);
    --el;
    uint Ml = (uint)BUILTIN_FLDEXP_F32(BUILTIN_ABS_F32(ml), 24);

    ulong S = AS_INT(x.lo) < 0 ? (win64(Mh, eh) - win64(Ml, el)) : (win64(Mh, eh) + win64(Ml, el));

    // Merged window is a single value: same tail as eptrigredlargeF.cl.
    uint p7 = (uint)(S >> 32);
    uint p6 = (uint)S;
    uint p5 = 0U;
    uint p4 = 0U;

    int i = p7 >> 29;

    p7 = BUILTIN_FSHR_B32(p7, p6, 30u);
    p6 = BUILTIN_FSHR_B32(p6, p5, 30u);
    p5 = BUILTIN_FSHR_B32(p5, p4, 30u);

    uint flip = i & 1 ? 0xffffffffU : 0U;
    uint sign = i & 1 ? (uint)SIGNBIT_SP32 : 0U;
    p7 = p7 ^ flip;
    p6 = p6 ^ flip;
    p5 = p5 ^ flip;

    int xe = BUILTIN_CLZ_U32(p7) + 1;
    uint shift = 32 - xe;
    p7 = BUILTIN_FSHR_B32(p7, p6, shift);
    p6 = BUILTIN_FSHR_B32(p6, p5, shift);

    float q1 = AS_FLOAT(sign | ((127 - xe) << 23) | (p7 >> 9));

    p7 = BUILTIN_FSHR_B32(p7, p6, 32u - 23u);
    int xxe = BUILTIN_CLZ_U32(p7) + 1;
    p7 = BUILTIN_FSHR_B32(p7, p6, 32u - xxe);
    float q0 = AS_FLOAT(sign | ((127 - (xe + 23 + xxe)) << 23) | (p7 >> 9));

    const float pio2h = (float)0xc90fda / 0x1.0p+23f;
    const float pio2hh = (float)0xc90 / 0x1.0p+11f;
    const float pio2ht = (float)0xfda / 0x1.0p+23f;
    const float pio2t = (float)0xa22168 / 0x1.0p+47f;

    float rh, rt;
    if (HAVE_FAST_FMA32() || !DAZ_OPT()) {
        rh = q1 * pio2h;
        rt = BUILTIN_FMA_F32(q0, pio2h, BUILTIN_FMA_F32(q1, pio2t, BUILTIN_FMA_F32(q1, pio2h, -rh)));
    } else {
        float q1h = AS_FLOAT(AS_UINT(q1) & 0xfffff000);
        float q1t = q1 - q1h;
        rh = q1 * pio2h;
        rt = MATH_MAD(q1t, pio2ht, MATH_MAD(q1t, pio2hh, MATH_MAD(q1h, pio2ht, MATH_MAD(q1h, pio2hh, -rh)))) +
             MATH_MAD(q0, pio2h, q1 * pio2t);
    }

    struct epredret ret;
    float t = rh + rt;
    ret.r.hi = t;
    ret.r.lo = rt - (t - rh);
    ret.i = ((i >> 1) + (i & 1)) & 0x3;
    return ret;
}
