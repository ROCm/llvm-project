// Objects compiled against the previous host-shadow offload-PGO runtime emit
// references to __llvm_profile_offload_register_* symbols. After the drain was
// decoupled (HSA introspection) these became empty no-ops kept only for ABI
// compatibility. Verify such references still link against libclang_rt.profile
// and run without effect. Referencing them force-links the host drain object,
// whose constructor dlopens HSA/HIP (absent here) and degrades gracefully.

// REQUIRES: linux

// RUN: %clang_profgen -o %t %s -ldl -lpthread
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t

void __llvm_profile_offload_register_shadow_variable(void *);
void __llvm_profile_offload_register_section_shadow_variable(void *);
void __llvm_profile_offload_register_dynamic_module(int, void **, const void *);
void __llvm_profile_offload_unregister_dynamic_module(void *);

int main(void) {
  int x = 0;
  void *mods[1] = {&x};
  __llvm_profile_offload_register_shadow_variable(&x);
  __llvm_profile_offload_register_section_shadow_variable(&x);
  __llvm_profile_offload_register_dynamic_module(1, mods, &x);
  __llvm_profile_offload_unregister_dynamic_module(&x);
  return 0;
}
