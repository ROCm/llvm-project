/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR double4
MATH_PRIVATE(epsincosep)(double2 x)
{
    double2 ax = x.hi < 0.0 ? -x : x;

    struct epredret r = MATH_PRIVATE(eptrigredep)(ax);
    double4 sc = MATH_PRIVATE(epsincosredep)(r.r);
    double2 cr = sc.lo;
    double2 sr = sc.hi;

    bool odd = (r.i & 1) != 0;
    double2 s = odd ? cr : sr;
    double2 c = odd ? -sr : cr;

    if (r.i > 1) {
        s = -s;
        c = -c;
    }

    long sgn = AS_LONG(x.hi) & SIGNBIT_DP64;
    s.lo = AS_DOUBLE(AS_LONG(s.lo) ^ sgn);
    s.hi = AS_DOUBLE(AS_LONG(s.hi) ^ sgn);

    return (double4)(c, s);
}
