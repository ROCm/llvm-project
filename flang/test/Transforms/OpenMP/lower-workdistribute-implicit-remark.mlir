// Verifies the user-visible remark emitted whenever a candidate `fir.do_loop
// unordered` is implicitly wrapped under `-fopenmp-implicit-workdistribute=
// host`, and that loops which are *not* candidates (ordered, already inside
// an OpenMP region, inner of a nested unordered pair, carrying iter_args, or
// carrying reduce operands) do not produce a remark.

// RUN: fir-opt --lower-workdistribute='implicit=host' --verify-diagnostics %s

func.func @wrapped(%lb : index, %ub : index, %step : index,
                   %addr : !fir.ref<index>) {
  // expected-remark@+1 {{implicit-workdistribute: wrapped array loop in `omp.teams { omp.workdistribute { ... } }`}}
  fir.do_loop %iv = %lb to %ub step %step unordered {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// Ordered loop: no remark.
func.func @no_remark_ordered(%lb : index, %ub : index, %step : index,
                              %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// Already inside an explicit workdistribute: no remark from the implicit pass.
func.func @no_remark_explicit(%lb : index, %ub : index, %step : index,
                               %addr : !fir.ref<index>) {
  omp.teams {
    omp.workdistribute {
      fir.do_loop %iv = %lb to %ub step %step unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr : !fir.ref<index>
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Loop carrying iter_args / producing a result is not a candidate.
// (Reductions and induction-style accumulators feed `workdistributeDoLower`
// a form it asserts against and would lose the produced value.) No remark.
func.func @no_remark_iter_args(%lb : index, %ub : index, %step : index,
                                %init : i32) -> i32 {
  %r = fir.do_loop %iv = %lb to %ub step %step unordered
      iter_args(%acc = %init) -> (i32) {
    fir.result %acc : i32
  }
  return %r : i32
}

// Inner loop of a nested unordered pair: only the outer is wrapped, so only
// one remark on the outer location.
func.func @one_remark_outer_only(%lb : index, %ub : index, %step : index,
                                  %addr : !fir.ref<index>) {
  // expected-remark@+1 {{implicit-workdistribute: wrapped array loop in `omp.teams { omp.workdistribute { ... } }`}}
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.do_loop %iv2 = %lb to %ub step %step unordered {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}
