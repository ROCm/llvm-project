module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.allocation_policy = #fir.allocation_policy<stack_arrays = false, small_array_threshold = 1024, total_stack_limit = 4194304>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.relocation_model = 1 : i32, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 24.0.0 (https://github.com/ROCm/llvm-project.git ffe5058289479b8ef6bb2188fa564bd138ed3032)", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.flags = #omp.flags<openmp_device_version = 61>, omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>, omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp.clause_requires<none>, omp.target_triples = [], omp.version = #omp.version<version = 61>} {
  omp.declare_mapper @_QMpolymorphic_nested_rtti_modTbase_t_omp_default_mapper : !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}> {
  ^bb0(%arg0: !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>):
    %0 = fir.declare %arg0 {uniq_name = ""} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    %1 = omp.map.info var_ptr(%0 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>) map_clauses(implicit, tofrom) capture(ByRef) members( :  : ) name("") -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    omp.declare_mapper.info map_entries(%1 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>)
  }
  omp.declare_mapper @_QQMpolymorphic_nested_rtti_modwrapper_t_omp_default_mapper : !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}> {
  ^bb0(%arg0: !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>):
    %0 = fir.declare %arg0 {uniq_name = ""} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    %1 = fir.coordinate_of %0, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %2 = fir.box_offset %1 base_addr : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>
    %3 = omp.map.info var_ptr(%1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) map_clauses(implicit, tofrom, ref_ptee) capture(ByRef) var_ptr_ptr(%2 : !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>, i8) mapper(@_QMpolymorphic_nested_rtti_modTbase_t_omp_default_mapper) name("") -> !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>
    %4 = omp.map.info var_ptr(%1 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>, !fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr(%2 : !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>, i8) name("") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %5 = omp.map.info var_ptr(%0 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>) map_clauses(implicit, tofrom) capture(ByRef) members(%3 : [1] : !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) name("") -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    omp.declare_mapper.info map_entries(%5, %3, %4 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>, !fir.llvm_ptr<!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>)
  }
  func.func @_QMpolymorphic_nested_rtti_modPinit_wrapper(%arg0: !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>> {fir.bindc_name = "w"}) {
    %c12_i32 = arith.constant 12 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %false = arith.constant false
    %c5_i32 = arith.constant 5 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMpolymorphic_nested_rtti_modFinit_wrapperEw"} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>, !fir.dscope) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    %2 = fir.embox %1 : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.box<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    %3 = fir.convert %2 : (!fir.box<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.box<none>
    fir.call @_FortranADestroy(%3) fastmath<contract> : (!fir.box<none>) -> ()
    %4 = fir.coordinate_of %1, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %5 = fir.zero_bits !fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    %6 = fir.embox %5 : (!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>
    fir.store %6 to %4 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %7 = fir.field_index marker, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %8 = fir.coordinate_of %1, marker : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<i32>
    fir.store %c5_i32 to %8 : !fir.ref<i32>
    %9 = fir.absent !fir.box<none>
    %10 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
    %11 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %12 = fir.coordinate_of %1, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %13 = fir.type_desc !fir.type<_QMpolymorphic_nested_rtti_modTchild_t{base_t:!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>,child_value:i32}>
    %14 = fir.convert %12 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>) -> !fir.ref<!fir.box<none>>
    %15 = fir.convert %13 : (!fir.tdesc<!fir.type<_QMpolymorphic_nested_rtti_modTchild_t{base_t:!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>,child_value:i32}>>) -> !fir.ref<none>
    fir.call @_FortranAAllocatableInitDerivedForAllocate(%14, %15, %c0_i32, %c0_i32) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
    %16 = fir.convert %12 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>) -> !fir.ref<!fir.box<none>>
    %17 = fir.convert %c-1_i64 : (i64) -> !fir.ref<i64>
    %18 = fir.convert %10 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %19 = fir.convert %false : (i1) -> !fir.llvm_ptr<(!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64) -> !fir.llvm_ptr<i8>>
    %20 = fir.call @_FortranAAllocatableAllocate(%16, %17, %false, %9, %18, %c32_i32, %19) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, !fir.llvm_ptr<(!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64) -> !fir.llvm_ptr<i8>>) -> i32
    %21 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %22 = fir.coordinate_of %1, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %23 = fir.load %22 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %24 = fir.field_index base_value, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>
    %25 = fir.coordinate_of %23, base_value : (!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.ref<i32>
    fir.store %c12_i32 to %25 : !fir.ref<i32>
    return
  }
  func.func @_QQmain() attributes {fir.bindc_name = "MAIN"} {
    %c79_i32 = arith.constant 79 : i32
    %c75_i32 = arith.constant 75 : i32
    %c13_i32 = arith.constant 13 : i32
    %c70_i32 = arith.constant 70 : i32
    %c65_i32 = arith.constant 65 : i32
    %c1_i32 = arith.constant 1 : i32
    %c28 = arith.constant 28 : index
    %c60_i32 = arith.constant 60 : i32
    %c6_i32 = arith.constant 6 : i32
    %true = arith.constant true
    %false = arith.constant false
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca !fir.logical<4> <{bindc_name = "base_extends_item", uniq_name = "_QFEbase_extends_item"}>
    %2 = fir.declare %1 {uniq_name = "_QFEbase_extends_item"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
    %3 = fir.alloca !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}> <{bindc_name = "base_mold", uniq_name = "_QFEbase_mold"}>
    %4 = fir.declare %3 {uniq_name = "_QFEbase_mold"} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    %5 = fir.alloca !fir.logical<4> <{bindc_name = "extends_base", uniq_name = "_QFEextends_base"}>
    %6 = fir.declare %5 {uniq_name = "_QFEextends_base"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
    %7 = fir.address_of(@_QFEw) : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    %8 = fir.declare %7 {uniq_name = "_QFEw"} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    fir.call @_QMpolymorphic_nested_rtti_modPinit_wrapper(%8) fastmath<contract> : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> ()
    %9 = fir.convert %false : (i1) -> !fir.logical<4>
    fir.store %9 to %6 : !fir.ref<!fir.logical<4>>
    %10 = fir.convert %true : (i1) -> !fir.logical<4>
    fir.store %10 to %2 : !fir.ref<!fir.logical<4>>
    %11 = omp.map.info var_ptr(%8 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>) map_clauses(tofrom) capture(ByRef) mapper(@_QQMpolymorphic_nested_rtti_modwrapper_t_omp_default_mapper) name("w") -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
    %12 = omp.map.info var_ptr(%6 : !fir.ref<!fir.logical<4>>, !fir.logical<4>) map_clauses(tofrom) capture(ByRef) name("extends_base") -> !fir.ref<!fir.logical<4>>
    %13 = omp.map.info var_ptr(%2 : !fir.ref<!fir.logical<4>>, !fir.logical<4>) map_clauses(tofrom) capture(ByRef) name("base_extends_item") -> !fir.ref<!fir.logical<4>>
    %14 = omp.map.info var_ptr(%4 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>) map_clauses(to) capture(ByRef) name("base_mold") -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    omp.target kernel_type(generic) map_entries(%11 -> %arg0, %12 -> %arg1, %13 -> %arg2, %14 -> %arg3 : !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>, !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) {
      %c1_i32_0 = arith.constant 1 : i32
      %true_1 = arith.constant true
      %64 = fir.declare %arg0 {uniq_name = "_QFEw"} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>
      %65 = fir.declare %arg1 {uniq_name = "_QFEextends_base"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
      %66 = fir.declare %arg2 {uniq_name = "_QFEbase_extends_item"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
      %67 = fir.declare %arg3 {uniq_name = "_QFEbase_mold"} : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
      %68 = fir.convert %true_1 : (i1) -> !fir.logical<4>
      fir.store %68 to %65 : !fir.ref<!fir.logical<4>>
      %69 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
      %70 = fir.coordinate_of %64, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %71 = fir.embox %67 : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.box<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
      %72 = fir.load %70 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %73 = fir.convert %71 : (!fir.box<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.box<none>
      %74 = fir.convert %72 : (!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.box<none>
      %75 = fir.call @_FortranAExtendsTypeOf(%73, %74) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
      %76 = fir.convert %75 : (i1) -> !fir.logical<4>
      fir.store %76 to %66 : !fir.ref<!fir.logical<4>>
      %77 = fir.field_index marker, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
      %78 = fir.coordinate_of %64, marker : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<i32>
      %79 = fir.load %78 : !fir.ref<i32>
      %80 = arith.addi %79, %c1_i32_0 : i32
      %81 = fir.field_index marker, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
      %82 = fir.coordinate_of %64, marker : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<i32>
      fir.store %80 to %82 : !fir.ref<i32>
      %83 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
      %84 = fir.coordinate_of %64, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %85 = fir.load %84 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %86 = fir.field_index base_value, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>
      %87 = fir.coordinate_of %85, base_value : (!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.ref<i32>
      %88 = fir.load %87 : !fir.ref<i32>
      %89 = arith.addi %88, %c1_i32_0 : i32
      %90 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
      %91 = fir.coordinate_of %64, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %92 = fir.load %91 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
      %93 = fir.field_index base_value, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>
      %94 = fir.coordinate_of %92, base_value : (!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.ref<i32>
      fir.store %89 to %94 : !fir.ref<i32>
      omp.terminator
    }
    scf.execute_region no_inline {
      %64 = fir.load %6 : !fir.ref<!fir.logical<4>>
      %65 = fir.convert %64 : (!fir.logical<4>) -> i1
      %66 = arith.xori %65, %true : i1
      cf.cond_br %66, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %67 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
      %68 = fir.convert %67 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
      %69 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %68, %c60_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
      %70 = fir.address_of(@_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D) : !fir.ref<!fir.char<1,28>>
      %71 = fir.declare %70 typeparams %c28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D"} : (!fir.ref<!fir.char<1,28>>, index) -> !fir.ref<!fir.char<1,28>>
      %72 = fir.convert %71 : (!fir.ref<!fir.char<1,28>>) -> !fir.ref<i8>
      %73 = fir.convert %c28 : (index) -> i64
      %74 = fir.call @_FortranAioOutputAscii(%69, %72, %73) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
      %75 = fir.call @_FortranAioEndIoStatement(%69) fastmath<contract> : (!fir.ref<i8>) -> i32
      fir.call @_FortranAStopStatement(%c1_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
      fir.unreachable
    ^bb2:  // pred: ^bb0
      scf.yield
    }
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %15 = fir.load %2 : !fir.ref<!fir.logical<4>>
    %16 = fir.convert %15 : (!fir.logical<4>) -> i1
    cf.cond_br %16, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %17 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
    %18 = fir.convert %17 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %19 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %18, %c65_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %20 = fir.address_of(@_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D) : !fir.ref<!fir.char<1,28>>
    %21 = fir.declare %20 typeparams %c28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D"} : (!fir.ref<!fir.char<1,28>>, index) -> !fir.ref<!fir.char<1,28>>
    %22 = fir.convert %21 : (!fir.ref<!fir.char<1,28>>) -> !fir.ref<i8>
    %23 = fir.convert %c28 : (index) -> i64
    %24 = fir.call @_FortranAioOutputAscii(%19, %22, %23) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %25 = fir.call @_FortranAioEndIoStatement(%19) fastmath<contract> : (!fir.ref<i8>) -> i32
    fir.call @_FortranAStopStatement(%c1_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
    fir.unreachable
  ^bb3:  // pred: ^bb1
    %26 = fir.field_index marker, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %27 = fir.coordinate_of %8, marker : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<i32>
    %28 = fir.load %27 : !fir.ref<i32>
    %29 = arith.cmpi ne, %28, %c6_i32 : i32
    cf.cond_br %29, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %30 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
    %31 = fir.convert %30 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %32 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %31, %c70_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %33 = fir.address_of(@_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D) : !fir.ref<!fir.char<1,28>>
    %34 = fir.declare %33 typeparams %c28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D"} : (!fir.ref<!fir.char<1,28>>, index) -> !fir.ref<!fir.char<1,28>>
    %35 = fir.convert %34 : (!fir.ref<!fir.char<1,28>>) -> !fir.ref<i8>
    %36 = fir.convert %c28 : (index) -> i64
    %37 = fir.call @_FortranAioOutputAscii(%32, %35, %36) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %38 = fir.call @_FortranAioEndIoStatement(%32) fastmath<contract> : (!fir.ref<i8>) -> i32
    fir.call @_FortranAStopStatement(%c1_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
    fir.unreachable
  ^bb5:  // pred: ^bb3
    %39 = fir.field_index item, !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %40 = fir.coordinate_of %8, item : (!fir.ref<!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %41 = fir.load %40 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>>
    %42 = fir.field_index base_value, !fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>
    %43 = fir.coordinate_of %41, base_value : (!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.ref<i32>
    %44 = fir.load %43 : !fir.ref<i32>
    %45 = arith.cmpi ne, %44, %c13_i32 : i32
    cf.cond_br %45, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %46 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
    %47 = fir.convert %46 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %48 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %47, %c75_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %49 = fir.address_of(@_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D) : !fir.ref<!fir.char<1,28>>
    %50 = fir.declare %49 typeparams %c28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D"} : (!fir.ref<!fir.char<1,28>>, index) -> !fir.ref<!fir.char<1,28>>
    %51 = fir.convert %50 : (!fir.ref<!fir.char<1,28>>) -> !fir.ref<i8>
    %52 = fir.convert %c28 : (index) -> i64
    %53 = fir.call @_FortranAioOutputAscii(%48, %51, %52) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %54 = fir.call @_FortranAioEndIoStatement(%48) fastmath<contract> : (!fir.ref<i8>) -> i32
    fir.call @_FortranAStopStatement(%c1_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> ()
    fir.unreachable
  ^bb7:  // pred: ^bb5
    %55 = fir.address_of(@_QQclX39f0aba7b4e4120a8dc1aa199e3bd119) : !fir.ref<!fir.char<1,71>>
    %56 = fir.convert %55 : (!fir.ref<!fir.char<1,71>>) -> !fir.ref<i8>
    %57 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %56, %c79_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %58 = fir.address_of(@_QQclX3D3D3D3D3D3D3D20546573742050617373656421203D3D3D3D3D3D3D) : !fir.ref<!fir.char<1,28>>
    %59 = fir.declare %58 typeparams %c28 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3D3D3D3D3D3D3D20546573742050617373656421203D3D3D3D3D3D3D"} : (!fir.ref<!fir.char<1,28>>, index) -> !fir.ref<!fir.char<1,28>>
    %60 = fir.convert %59 : (!fir.ref<!fir.char<1,28>>) -> !fir.ref<i8>
    %61 = fir.convert %c28 : (index) -> i64
    %62 = fir.call @_FortranAioOutputAscii(%57, %60, %61) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %63 = fir.call @_FortranAioEndIoStatement(%57) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  func.func private @_FortranADestroy(!fir.box<none>) attributes {fir.runtime}
  fir.global linkonce @_QQclX39f0aba7b4e4120a8dc1aa199e3bd119 constant : !fir.char<1,71> {
    %0 = fir.string_lit "offload/test/offloading/fortran/target-map-polymorphic-nested-rtti.f90\00"(71) : !fir.char<1,71>
    fir.has_value %0 : !fir.char<1,71>
  }
  func.func private @_FortranAAllocatableInitDerivedForAllocate(!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) attributes {fir.runtime}
  func.func private @_FortranAAllocatableAllocate(!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, !fir.llvm_ptr<(!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64) -> !fir.llvm_ptr<i8>>) -> i32 attributes {fir.runtime}
  fir.global internal @_QFEw : !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}> {
    %0 = fir.undefined !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %1 = fir.zero_bits i32
    %2 = fir.insert_value %0, %1, ["marker", !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>] : (!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>, i32) -> !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    %3 = fir.zero_bits !fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>
    %4 = fir.embox %3 : (!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>
    %5 = fir.insert_value %2, %4, ["item", !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>] : (!fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>, !fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>) -> !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
    fir.has_value %5 : !fir.type<_QMpolymorphic_nested_rtti_modTwrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<_QMpolymorphic_nested_rtti_modTbase_t{base_value:i32}>>>}>
  }
  func.func private @_FortranAExtendsTypeOf(!fir.box<none>, !fir.box<none>) -> i1 attributes {fir.runtime}
  func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  func.func private @_FortranAioOutputAscii(!fir.ref<i8>, !fir.ref<i8>, i64) -> i1 attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclX3D3D3D3D3D3D3D2054657374204661696C656421203D3D3D3D3D3D3D constant : !fir.char<1,28> {
    %0 = fir.string_lit "======= Test Failed! ======="(28) : !fir.char<1,28>
    fir.has_value %0 : !fir.char<1,28>
  }
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
  func.func private @_FortranAStopStatement(i32, i1, i1) attributes {fir.runtime}
  fir.global linkonce @_QQclX3D3D3D3D3D3D3D20546573742050617373656421203D3D3D3D3D3D3D constant : !fir.char<1,28> {
    %0 = fir.string_lit "======= Test Passed! ======="(28) : !fir.char<1,28>
    fir.has_value %0 : !fir.char<1,28>
  }
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
    fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) fastmath<contract> : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) -> ()
    fir.call @_QQmain() fastmath<contract> : () -> ()
    fir.call @_FortranAProgramEndStatement() fastmath<contract> : () -> ()
    return %c0_i32 : i32
  }
}
