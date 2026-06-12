// COM: Test Comgr parse-isa-name() API
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack-" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:sramecc-" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack+:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack-:sramecc+" SUCCESS
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx908:xnack-:sramecc-" SUCCESS
// RUN: parse-isa-name "spirv64-amd-amdhsa--amdgcnspirv" SUCCESS
// RUN: parse-isa-name "spirv64-amd-amdhsa-unknown-amdgcnspirv" SUCCESS

// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx1010:xnack+" SUCCESS
// RUN: parse-isa-name "" SUCCESS

// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx801:xnack+:sramecc+" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803:::" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa-opencl-gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa-gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name "gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgcn-amd-amdhsa--gfx803" INVALID_ARGUMENT
// RUN: parse-isa-name " amdgcn-amd-amdhsa--gfx803 " INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803 " INVALID_ARGUMENT
// RUN: parse-isa-name "   amdgcn-amd-amdhsa--gfx803  " INVALID_ARGUMENT
// RUN: parse-isa-name "amdgcn-amd-amdhsa--gfx803  " INVALID_ARGUMENT
// RUN: parse-isa-name "spirv64-amd-amdhsa--amdgcnspirv:xnack+" INVALID_ARGUMENT
