// Simple HIP kernel for testing kernel cloning with different block sizes

extern "C" {

__attribute__((global))
void kernel1(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(1, 256)))
void kernel2(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(256, 1024)))
void kernel3(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(128, 768)))
void kernel4(float* a) {
    a[0] = 1.0f;
}

__attribute__((global))
__attribute__((amdgpu_flat_work_group_size(512, 512)))
void kernel5(float* a) {
    a[0] = 1.0f;
}

}
