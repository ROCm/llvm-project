/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCML_REDRETH_H
#define OCML_REDRETH_H

// Single-part reduced argument: x = i*(pi/2) + hi, |hi| <= pi/4.
struct redret {
    half hi;
    short i;
};

struct scret {
    half s;
    half c;
};

#endif // OCML_REDRETH_H
