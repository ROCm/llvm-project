
#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(erfcx)(float x)
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
    e = BUILTIN_FMA_F32(-q, tx, 1.0f) + (p - q);
    q = BUILTIN_FMA_F32(r, e, q);
    return q;
}

#if !defined EXTRA_ACCURACY

CONSTATTR float
MATH_MANGLE(erfcx)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;
    
    if (ax < 0x1.41bbf8p+3f) {
        ret = MATH_PRIVATE(erfcx)(ax);
    } else {
        float r = MATH_FAST_RCP(0x1.0p-2f * ax);
        float t = r*r * 0x1.0p-4f;
        float p = PE4(t, 6.5625f, -1.875f, 0.75f, -0.5f, 1.0f);
        ret = 0x1.20dd76p-3f * r * p;
    }

    if (x < 0.0f) {
        float x2h = x*x;
        float x2l = BUILTIN_FMA_F32(x, x, -x2h);
        float e = MATH_MANGLE(exp)(x2h);
        ret = BUILTIN_FMA_F32(2.0f, BUILTIN_FMA_F32(e, x2l, e), -ret);
        ret = x < -0x1.2d6abcp+3f ? PINF_F32 : ret;
    }

    return ret;
}

#else

CONSTATTR float
MATH_MANGLE(erfcx)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 1.0f) {
        ret = PE13(x, -0x1.77d64p-11f, 0x1.269372p-9f, -0x1.c27dd4p-9f, 0x1.d3d3c4p-8f, -0x1.35d6cap-6f,
                    0x1.5bb082p-5f, -0x1.60e46ep-4f, 0x1.54d3e4p-3f, -0x1.340edap-2f,
                    0x1.00049ap-1f, -0x1.81286p-1f, 0x1.ffffcap-1f, -0x1.20dd7p+0f, 0x1.0p+0f);
    } else if (ax < 32.0f) {
        float n = ax - 4.0f;
        float d = ax + 4.0f;
        float r = MATH_FAST_RCP(d);
        float t = n * r;
        float e = BUILTIN_FMA_F32(-t, ax, BUILTIN_FMA_F32(t + 1.0f, -4.0f, ax));
        t = BUILTIN_FMA_F32(r, e, t);
        float p = PE9(t, 0.00416076401f, -0.0167250745f, 0.0378070959f, -0.0661972834f, 0.0935599947f, -0.101052745f,
                        0.0681148962f, 0.0153801711f, -0.139621619f, 0x1.dd2c9p-3f);

        float tx = ax + ax;
        d = 1.0f + tx;
        r = MATH_FAST_RCP(d);
        float q = BUILTIN_FMA_F32(p, r, r);
        e = BUILTIN_FMA_F32(-q, tx, 1.0f) + (p - q);
        ret = BUILTIN_FMA_F32(r, e, q);
    } else {
        const float one_over_sqrtpi = 0x1.20dd76p-1f;
        float z = MATH_RCP(x * x);
        ret =  MATH_DIV(one_over_sqrtpi, x) * PE2(z, 0.375f, -0.5f, 1.0f);
    }

    if (x <= -1.0f) {
        float x2h, x2l;
        if (HAVE_FAST_FMA32()) {
            x2h = ax * ax;
            x2l = BUILTIN_FMA_F32(ax, ax, -x2h);
        } else {
            float xh = AS_FLOAT(AS_UINT(ax) & 0xfffff000U);
            float xl = ax - xh;
            x2h = xh*xh;
            x2l = (ax + xh)*xl;
        }

        ret = MATH_MANGLE(exp)(x2h) * MATH_MANGLE(exp)(x2l) * 2.0f - ret;
        ret = x < -10.0f ? PINF_F32 : ret;
    }

    return ret;
}

#endif
