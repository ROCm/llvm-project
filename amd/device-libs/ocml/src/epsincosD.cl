/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR double4
MATH_PRIVATE(epsincos)(double y)
{
    double ay = BUILTIN_ABS_F64(y);
    if (!FINITE_ONLY_OPT())
        ay = BUILTIN_ISINF_F64(ay) ? QNAN_F64 : ay;

    struct redret r = MATH_PRIVATE(trigred)(ay);
    double4 sc = MATH_PRIVATE(epsincosred2)(r.hi, r.lo);
    double2 cr = sc.lo;
    double2 sr = sc.hi;

    bool odd = (r.i & 1) != 0;
    double2 s = odd ? cr : sr;
    double2 c = odd ? -sr : cr;

    if (r.i > 1) {
        s = -s;
        c = -c;
    }

    long sgn = AS_LONG(y) & SIGNBIT_DP64;
    s.lo = AS_DOUBLE(AS_LONG(s.lo) ^ sgn);
    s.hi = AS_DOUBLE(AS_LONG(s.hi) ^ sgn);

    return (double4)(c, s);
}
