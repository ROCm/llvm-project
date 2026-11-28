/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCML_REDRETD_H
#define OCML_REDRETD_H

// Single-part reduced argument: x = i*(pi/2) + hi, |hi| <= pi/4.
struct redret {
    double hi;
    int i;
};

// Extra-precision reduced argument: x = i*(pi/2) + (r.hi + r.lo), |r.hi + r.lo| <= pi/4.
struct epredret {
    double2 r;
    int i;
};

struct scret {
    double s;
    double c;
};

#endif // OCML_REDRETD_H
