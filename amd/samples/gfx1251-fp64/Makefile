# make        build ./fp64
# make run    run the kernels on a gfx1251 GPU and check results
# make check  verify every instruction named in fp64.cpp is in the gfx1251 ISA
HIPCC ?= hipcc
ARCH  ?= gfx1251
FLAGS := -x hip -O3 --offload-arch=$(ARCH) -munsafe-fp-atomics

fp64: fp64.cpp
	$(HIPCC) $(FLAGS) $< -o $@

fp64-$(ARCH).s: fp64.cpp
	$(HIPCC) $(FLAGS) --cuda-device-only -S $< -o $@

run: fp64
	./fp64

check: fp64-$(ARCH).s
	@n=0; miss=0; \
	for i in $$(grep -oE '\b(v|ds|global|flat|buffer)_[a-z0-9_]*f64[a-z0-9_]*' fp64.cpp | sort -u); do \
	  n=$$((n + 1)); \
	  grep -qwE "$$i(_e32|_e64|_dpp|_e64_dpp)?" $< || { echo "missing: $$i"; miss=$$((miss + 1)); }; \
	done; \
	echo "$$((n - miss))/$$n instructions found in $<"; [ $$miss -eq 0 ]

clean:
	rm -f fp64 fp64-*.s

.PHONY: run check clean
