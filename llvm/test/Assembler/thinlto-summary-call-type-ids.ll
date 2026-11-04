; Test parsing and printing of call type identifiers in function summaries.
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

^0 = module: (path: "thinlto-summary-call-type-ids.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, callTypeId: 13568473337159563479, indirectCallTypeIds: (18428409508369012832, 11966876714456557523))))
; Only the identifier of the function.
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, callTypeId: 10830334733921094663)))
; Only the identifiers of indirect calls, before other fields.
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 3, indirectCallTypeIds: (657395674655738670), calls: ((callee: ^2)))))

; CHECK: ^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 2, callTypeId: 13568473337159563479, indirectCallTypeIds: (18428409508369012832, 11966876714456557523))))
; CHECK: ^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1, callTypeId: 10830334733921094663)))
; CHECK: ^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 3, calls: ((callee: ^2)), indirectCallTypeIds: (657395674655738670))))
