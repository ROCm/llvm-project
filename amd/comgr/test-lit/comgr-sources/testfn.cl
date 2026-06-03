// Standard
// clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
//  -c testfn.cl -o testfn.o
// clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
//  -shared testfn.cl -o testfn.so

__attribute__((visibility("default"))) constant int foo = 0;

void kernel testfn(
    global int *a, const global int *b) {
  *a = *b;
}
