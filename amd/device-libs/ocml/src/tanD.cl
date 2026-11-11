/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR double
MATH_MANGLE(tan)(double x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F64(x) ? QNAN_F64 : x;

    double ax = BUILTIN_ABS_F64(x);

    struct epredret r = MATH_PRIVATE(eptrigred)(ax);

    double t = MATH_PRIVATE(tanredep)(r.r, r.i & 1);
    t = AS_DOUBLE(AS_LONG(t) ^ (AS_LONG(x) & SIGNBIT_DP64));

    return AS_DOUBLE(t);
}

