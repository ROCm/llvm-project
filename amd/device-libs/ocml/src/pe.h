/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Polynomial evaluation by Horner's rule, built on MATH_MAD.
//
//   PEn(X, Cn, C(n-1), ..., C1, C0)  ==  Cn*X^n + ... + C1*X + C0
//
// Each level expands to the same nested MATH_MAD chain that was previously
// written out by hand, so conversion is bit-identical.  The evaluation scheme
// lives here alone: changing it (e.g. an even/odd or Estrin split to shorten
// the dependency chain) means editing this file, not every call site.
//
// This header is generated; regenerate rather than editing by hand.

#ifndef OCML_PE_H
#define OCML_PE_H

#define PE1(X, C1, C0) MATH_MAD(X, C1, C0)
#define PE2(X, C2, C1, C0) MATH_MAD(X, PE1(X, C2, C1), C0)
#define PE3(X, C3, C2, C1, C0) MATH_MAD(X, PE2(X, C3, C2, C1), C0)
#define PE4(X, C4, C3, C2, C1, C0) MATH_MAD(X, PE3(X, C4, C3, C2, C1), C0)
#define PE5(X, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE4(X, C5, C4, C3, C2, C1), C0)
#define PE6(X, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE5(X, C6, C5, C4, C3, C2, C1), C0)
#define PE7(X, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE6(X, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE8(X, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE7(X, C8, C7, C6, C5, C4, C3, C2, \
        C1), C0)
#define PE9(X, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE8(X, C9, C8, C7, C6, C5, C4, \
        C3, C2, C1), C0)
#define PE10(X, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE9(X, C10, C9, C8, C7, \
        C6, C5, C4, C3, C2, C1), C0)
#define PE11(X, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE10(X, C11, C10, \
        C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE12(X, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE11(X, C12, \
        C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE13(X, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE12(X, \
        C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE14(X, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, \
        PE13(X, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE15(X, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, \
        PE14(X, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE16(X, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) \
        MATH_MAD(X, PE15(X, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, \
        C1), C0)
#define PE17(X, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) \
        MATH_MAD(X, PE16(X, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, \
        C2, C1), C0)
#define PE18(X, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, \
        C0) MATH_MAD(X, PE17(X, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, \
        C4, C3, C2, C1), C0)
#define PE19(X, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, \
        C1, C0) MATH_MAD(X, PE18(X, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, \
        C6, C5, C4, C3, C2, C1), C0)
#define PE20(X, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, \
        C2, C1, C0) MATH_MAD(X, PE19(X, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, \
        C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE21(X, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, \
        C4, C3, C2, C1, C0) MATH_MAD(X, PE20(X, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, \
        C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE22(X, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, \
        C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE21(X, C22, C21, C20, C19, C18, C17, C16, C15, C14, \
        C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE23(X, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, \
        C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE22(X, C23, C22, C21, C20, C19, C18, C17, C16, \
        C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE24(X, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, \
        C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE23(X, C24, C23, C22, C21, C20, C19, C18, \
        C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE25(X, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, \
        C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE24(X, C25, C24, C23, C22, C21, C20, \
        C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE26(X, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, \
        C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE25(X, C26, C25, C24, C23, C22, \
        C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, \
        C2, C1), C0)
#define PE27(X, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, \
        C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE26(X, C27, C26, C25, C24, \
        C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, \
        C4, C3, C2, C1), C0)
#define PE28(X, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, \
        C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE27(X, C28, C27, C26, \
        C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, C9, C8, \
        C7, C6, C5, C4, C3, C2, C1), C0)
#define PE29(X, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, \
        C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE28(X, C29, C28, \
        C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, C11, C10, \
        C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE30(X, C30, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, \
        C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE29(X, C30, \
        C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, C13, C12, \
        C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE31(X, C31, C30, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, \
        C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, PE30(X, \
        C31, C30, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, C16, C15, C14, \
        C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)
#define PE32(X, C32, C31, C30, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, \
        C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1, C0) MATH_MAD(X, \
        PE31(X, C32, C31, C30, C29, C28, C27, C26, C25, C24, C23, C22, C21, C20, C19, C18, C17, \
        C16, C15, C14, C13, C12, C11, C10, C9, C8, C7, C6, C5, C4, C3, C2, C1), C0)

#endif // OCML_PE_H
