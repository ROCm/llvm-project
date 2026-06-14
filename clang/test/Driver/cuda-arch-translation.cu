// Tests that "sm_XX" gets correctly converted to "compute_YY" when we invoke
// fatbinary.

// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_20 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM20 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_21 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM21 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_30 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM30 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_32 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM32 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_35 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM35 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_37 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM37 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_50 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM50 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_52 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM52 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_53 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM53 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM60 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_61 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM61 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_62 --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM62 %s
// RUN: %clang -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=sm_70 --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CUDA,SM70 %s

// CUDA: ptxas
// CUDA-SAME: -m64
// CUDA: fatbinary

// HIP: clang-offload-bundler

// SM20:--image3=kind=elf,sm=20{{.*}}
// SM21:--image3=kind=elf,sm=21{{.*}}
// SM30:--image3=kind=elf,sm=30{{.*}}
// SM32:--image3=kind=elf,sm=32{{.*}}
// SM35:--image3=kind=elf,sm=35{{.*}}
// SM37:--image3=kind=elf,sm=37{{.*}}
// SM50:--image3=kind=elf,sm=50{{.*}}
// SM52:--image3=kind=elf,sm=52{{.*}}
// SM53:--image3=kind=elf,sm=53{{.*}}
// SM60:--image3=kind=elf,sm=60{{.*}}
// SM61:--image3=kind=elf,sm=61{{.*}}
// SM62:--image3=kind=elf,sm=62{{.*}}
// SM70:--image3=kind=elf,sm=70{{.*}}
// GFX600:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx600
// GFX601:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx601
// GFX602:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx602
// GFX700:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx700
// GFX701:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx701
// GFX702:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx702
// GFX703:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx703
// GFX704:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx704
// GFX705:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx705
// GFX801:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx801
// GFX802:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx802
// GFX803:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx803
// GFX805:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx805
// GFX810:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx810
// GFX900:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx900
// GFX902:-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx902
// SPIRV:-targets=host-x86_64-unknown-linux-gnu,hip-spirv64-amd-amdhsa--amdgcnspirv
