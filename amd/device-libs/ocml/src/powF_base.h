/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epln)(float);
extern CONSTATTR float MATH_PRIVATE(expep)(float2);

static float
fast_expylnx(float x, float y)
{
    float ax = BUILTIN_ABS_F32(x);
    return BUILTIN_EXP2_F32(y * BUILTIN_LOG2_F32(ax));
}

// exp(y*ln(ax)) for y = yh + yl; the product is left unnormalized
// since expep only needs the sum
static float
expylnx2(float ax, float yh, float yl)
{
    float2 l = MATH_PRIVATE(epln)(ax);
    float ph = yh * l.hi;
    float pl = BUILTIN_FMA_F32(yh, l.lo, BUILTIN_FMA_F32(yl, l.hi, BUILTIN_FMA_F32(yh, l.hi, -ph)));
    return MATH_PRIVATE(expep)(con(ph, pl));
}

static float
compute_expylnx_int(float x, int ny)
{
    if (UNSAFE_MATH_OPT())
        return fast_expylnx(x, (float)ny);

    float ax = BUILTIN_ABS_F32(x);
    int nyh = ny & 0xffff0000;
    float2 y = fadd((float)nyh, (float)(ny - nyh));
    return expylnx2(ax, y.hi, y.lo);
}

// root version of compute_expylnx_int
static float
compute_exp_inverse_y_lnx_int(float x, int ny)
{
    float ax = BUILTIN_ABS_F32(x);
    if (UNSAFE_MATH_OPT()) {
        float y = MATH_FAST_RCP((float)ny);
        return fast_expylnx(ax, y);
    }

    int nyh = ny & 0xffff0000;
    float2 y = fadd((float)nyh, (float)(ny - nyh));
    float qh = MATH_FAST_RCP(y.hi);
    float ql = qh * BUILTIN_FMA_F32(-qh, y.lo, BUILTIN_FMA_F32(-qh, y.hi, 1.0f));
    return expylnx2(ax, qh, ql);
}

static float
compute_expylnx_float(float x, float y)
{
    if (UNSAFE_MATH_OPT())
        return fast_expylnx(x, y);

    float ax = BUILTIN_ABS_F32(x);
    float2 l = MATH_PRIVATE(epln)(ax);
    // y*l, left unnormalized since expep only needs the sum
    float ph = y * l.hi;
    float pl = BUILTIN_FMA_F32(y, l.lo, BUILTIN_FMA_F32(y, l.hi, -ph));
    return MATH_PRIVATE(expep)(con(ph, pl));
}

static bool is_integer(float ay)
{
    return BUILTIN_TRUNC_F32(ay) == ay;
}

static bool is_even_integer(float ay) {
    // Even integers are still integers after division by 2.
    return is_integer(0.5f * ay);
}

static bool is_odd_integer(float ay) {
    return is_integer(ay) && !is_even_integer(ay);
}

#if defined(COMPILING_POW)

CONSTATTR
static float
pow_fixup(float x, float y, float yin, float expylnx)
{
    float ax = BUILTIN_ABS_F32(x);
    bool is_odd_y = is_odd_integer(y);

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (x < 0.0f && !is_integer(y))
        ret = QNAN_F32;

    if (BUILTIN_ISINF_F32(ax) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (y < 0.0f) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);

    if (BUILTIN_ISUNORDERED_F32(x, yin))
        ret = QNAN_F32;

    return ret;
}

CONSTATTR float
MATH_MANGLE(pow)(float x, float y)
{
    if (x == 1.0f)
        y = 1.0f;
    if (y == 0.0f)
        x = 1.0f;

    // |y| >= 2^100 gives the same results as y = +-inf, including
    // pow(-1, +-inf) = 1, so infinite y needs no special case
    float yc = BUILTIN_CLAMP_F32(y, -0x1.0p+100f, 0x1.0p+100f);
    float expylnx = compute_expylnx_float(x, yc);

    return pow_fixup(x, yc, y, expylnx);
}

#elif defined(COMPILING_POWR)

CONSTATTR
static float
powr_fixup(float x, float y, float expylnx)
{
    float ret = expylnx;

    // Now all the edge cases
    float iz = y < 0.0f ? PINF_F32 : 0.0f;
    float zi = y < 0.0f ? 0.0f : PINF_F32;

    if (x == 0.0f)
        ret = y == 0.0f ? QNAN_F32 : iz;

    if (x == PINF_F32 && y != 0.0f)
        ret = zi;

    if (BUILTIN_ISUNORDERED_F32(x, y))
        ret = QNAN_F32;

    return ret;
}

CONSTATTR float
MATH_MANGLE(powr)(float x, float y)
{
    if (x < 0.0f)
        x = QNAN_F32;

    float expylnx = compute_expylnx_float(x, y);
    return powr_fixup(x, y, expylnx);
}

#elif defined(COMPILING_POWN)

CONSTATTR
static float
pown_fixup(float x, int ny, float expylnx)
{
    bool is_odd_y = ny & 1;

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F32(x) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);
    return ret;
}

CONSTATTR float
MATH_MANGLE(pown)(float x, int ny)
{
    if (ny == 0)
        x = 1.0f;

    float expylnx = compute_expylnx_int(x, ny);
    return pown_fixup(x, ny, expylnx);
}

#elif defined(COMPILING_ROOTN)

CONSTATTR
static float
rootn_fixup(float x, int ny, float expylnx)
{
    bool is_odd_y = ny & 1;

    float ret = BUILTIN_COPYSIGN_F32(expylnx, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F32(x) || x == 0.0f)
        ret = BUILTIN_COPYSIGN_F32((x == 0.0f) ^ (ny < 0) ? 0.0f : PINF_F32,
                                   is_odd_y ? x : 0.0f);

    if ((x < 0.0f && !is_odd_y) || ny == 0)
        ret = QNAN_F32;

    return ret;
}

CONSTATTR float
MATH_MANGLE(rootn)(float x, int ny)
{
    float expylnx = compute_exp_inverse_y_lnx_int(x, ny);
    return rootn_fixup(x, ny, expylnx);
}

#else
#error missing function macro
#endif

