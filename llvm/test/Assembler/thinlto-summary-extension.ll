;; Test the summary extensions in the summary assembly and in the bitcode of a
;; combined index.

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/index.ll -o %t/index.bc
; RUN: llvm-dis %t/index.bc -o - | FileCheck %s
; RUN: llvm-bcanalyzer -dump %t/index.bc | FileCheck %s --check-prefix=BCAN
;; A tool that does not build the index skips the extensions.
; RUN: opt -S %t/index.ll -o /dev/null

;; Each entry belongs to the summary of a value in one module. The two copies
;; of GUID 2 (for example, a template function in two translation units) have
;; different entries. The entries are in the order of the summaries. An index
;; can hold two versions of an extension. An extension without entries is not
;; written to bitcode.
; CHECK:      ^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
; CHECK-NEXT: ^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
; CHECK-NEXT: ^2 = gv: (guid: 1,
; CHECK-NEXT: ^3 = gv: (guid: 2,
; CHECK-NEXT: ^4 = extension: (name: "test.facts", version: 1, required: 0, entries: ((gv: ^3, module: ^0, ints: (1))))
; CHECK-NEXT: ^5 = extension: (name: "test.facts", version: 2, required: 0, entries: ((gv: ^2, module: ^0, ints: (1, 18446744073709551615)), (gv: ^3, module: ^0, ints: (3), strings: ("x", "", "y\22z")), (gv: ^3, module: ^1, ints: (4))))
; CHECK-NEXT: ^6 = extension: (name: "test.results", version: 1, required: 1, entries: ((gv: ^3, module: ^1, strings: ("attr"))))
; CHECK-NEXT: ^7 = blockcount: 0

; BCAN:      <SUMMARY_EXTENSION_BLOCK
; BCAN-NEXT:   <HEADER op0={{[0-9]+}} op1=10 op2=1 op3=0/>
; BCAN-NEXT:   <COMBINED_ENTRY op0=[[GUID2:[0-9]+]] op1=[[A:[0-9]+]] op2=1 op3=1/>
; BCAN-NEXT: </SUMMARY_EXTENSION_BLOCK>
; BCAN-NEXT: <SUMMARY_EXTENSION_BLOCK
; BCAN-NEXT:   <HEADER op0={{[0-9]+}} op1=10 op2=2 op3=0/>
; BCAN-NEXT:   <COMBINED_ENTRY op0={{[0-9]+}} op1=[[A]] op2=2 op3=1 op4=-1/>
; BCAN-NEXT:   <COMBINED_ENTRY op0=[[GUID2]] op1=[[A]] op2=1 op3=3 op4={{[0-9]+}} op5=1 op6={{[0-9]+}} op7=0 op8={{[0-9]+}} op9=3/>
; BCAN-NEXT:   <COMBINED_ENTRY op0=[[GUID2]] op1=[[B:[0-9]+]] op2=1 op3=4/>
; BCAN-NEXT: </SUMMARY_EXTENSION_BLOCK>
; BCAN-NEXT: <SUMMARY_EXTENSION_BLOCK
; BCAN-NEXT:   <HEADER op0={{[0-9]+}} op1=12 op2=1 op3=1/>
; BCAN-NEXT:   <COMBINED_ENTRY op0=[[GUID2]] op1=[[B]] op2=0 op3={{[0-9]+}} op4=4/>
; BCAN-NEXT: </SUMMARY_EXTENSION_BLOCK>

; RUN: not llvm-as %t/forward-ref.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FORWARD
; FORWARD: error: a summary extension entry must refer to an earlier summary entry

; RUN: not llvm-as %t/no-summary.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-SUMMARY
; NO-SUMMARY: error: the value has no summary in the module

; RUN: not llvm-as %t/bad-module.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-MODULE
; BAD-MODULE: error: unknown module ID

; RUN: not llvm-as %t/duplicate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE
; DUPLICATE: error: duplicate entry in summary extension 'test'

;--- index.ll
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1)))
^4 = extension: (name: "test.facts", version: 2, required: 0, entries: ((gv: ^3, module: ^1, ints: (4)), (gv: ^3, module: ^0, ints: (3), strings: ("x", "", "y\22z")), (gv: ^2, module: ^0, ints: (1, 18446744073709551615))))
^5 = extension: (name: "test.facts", version: 1, required: 0, entries: ((gv: ^3, module: ^0, ints: (1))))
^6 = extension: (name: "test.results", version: 1, required: 1, entries: ((gv: ^3, module: ^1, strings: ("attr"))))
^7 = extension: (name: "test.empty", version: 1, required: 0)

;--- forward-ref.ll
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = extension: (name: "test", version: 1, required: 0, entries: ((gv: ^2, module: ^0, ints: (1))))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))

;--- no-summary.ll
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^3 = extension: (name: "test", version: 1, required: 0, entries: ((gv: ^2, module: ^1, ints: (1))))

;--- bad-module.ll
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^2 = extension: (name: "test", version: 1, required: 0, entries: ((gv: ^1, module: ^1, ints: (1))))

;--- duplicate.ll
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^2 = extension: (name: "test", version: 1, required: 0, entries: ((gv: ^1, module: ^0, ints: (1)), (gv: ^1, module: ^0, ints: (2))))
