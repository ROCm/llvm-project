// Standard
// clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
//  -c gtgra.cl -o gtgra.o
// clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
//  -shared gtgra.cl -o gtgra.so

__attribute__((visibility("default"))) constant int foo = 0;

void kernel bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz(
    global int *a, const global int *b) {
  *a = *b;
}
