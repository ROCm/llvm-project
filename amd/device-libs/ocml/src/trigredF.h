/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define SMALL_BOUND 0x1.0p+17f

struct redret {
    float hi;
    int i;
};

// Extra-precision reduced argument: x = i*(pi/2) + (hi + lo), |hi + lo| <= pi/4.
struct redret2 {
    float hi;
    float lo;
    int i;
};

struct scret {
    float s;
    float c;
};

extern CONSTATTR struct redret MATH_PRIVATE(trigredsmall)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigredlarge)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigred)(float x);

extern CONSTATTR struct redret2 MATH_PRIVATE(trigred2small)(float x);
extern CONSTATTR struct redret2 MATH_PRIVATE(trigred2large)(float x);
extern CONSTATTR struct redret2 MATH_PRIVATE(trigred2)(float x);

extern CONSTATTR struct scret  MATH_PRIVATE(sincosred)(float x);
extern CONSTATTR struct scret  MATH_PRIVATE(sincosred2)(float x, float y);

// cos in .lo, sin in .hi
extern CONSTATTR float4 MATH_PRIVATE(epsincosred2)(float x, float y);
extern CONSTATTR float4 MATH_PRIVATE(epsincos)(float y);

extern CONSTATTR float MATH_PRIVATE(tanred)(float x, int regn);
extern CONSTATTR float MATH_PRIVATE(tanred2)(float x, float xx, int regn);

