/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCML_REDRETF_H
#define OCML_REDRETF_H

// Single-part reduced argument: x = i*(pi/2) + hi, |hi| <= pi/4.
struct redret {
    float hi;
    int i;
};

// Extra-precision reduced argument: x = i*(pi/2) + (r.hi + r.lo), |r.hi + r.lo| <= pi/4.
struct epredret {
    float2 r;
    int i;
};

struct scret {
    float s;
    float c;
};

#endif // OCML_REDRETF_H
