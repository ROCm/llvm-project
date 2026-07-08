/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

__ocml_sincos_f32_result
MATH_MANGLE(sincospi_stret)(float x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F32(x) ? QNAN_F32 : x;

    float ax = BUILTIN_ABS_F32(x);

    struct redret r = MATH_PRIVATE(trigpired)(ax);
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    int flip = r.i > 1 ? SIGNBIT_SP32 : 0;
    bool odd = (r.i & 1) != 0;
    float s = odd ? sc.c : sc.s;
    s = AS_FLOAT(AS_INT(s) ^ flip ^ (AS_INT(ax) ^ AS_INT(x)));
    sc.s = -sc.s;
    float c = odd ? sc.s : sc.c;
    c = AS_FLOAT(AS_INT(c) ^ flip);

    __ocml_sincos_f32_result result = {s, c};
    return result;
}

float
MATH_MANGLE(sincospi)(float x, __private float *cp)
{
    __ocml_sincos_f32_result result = MATH_MANGLE(sincospi_stret)(x);
    *cp = result.__cos;
    return result.__sin;
}
