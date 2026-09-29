/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(cos)(float x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F32(x) ? QNAN_F32 : x;

    float ax = BUILTIN_ABS_F32(x);

#if defined EXTRA_PRECISION
    struct epredret r = MATH_PRIVATE(eptrigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosredep)(r.r);
#else
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
#endif
    sc.s = -sc.s;

    float c =  (r.i & 1) != 0 ? sc.s : sc.c;
    c = AS_FLOAT(AS_INT(c) ^ (r.i > 1 ? SIGNBIT_SP32 : 0));

    return c;
}
