# -*- Python -*-

import os
import re


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal(
            "No attribute %r in test configuration! You may need to run "
            "tests from your build directory or add this attribute "
            "to lit.site.cfg.py " % attr_name
        )
    return attr_value


# Setup config name.
config.name = "Profile-" + config.target_arch

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup executable root.
if (
    hasattr(config, "profile_lit_binary_dir")
    and config.profile_lit_binary_dir is not None
):
    config.test_exec_root = os.path.join(config.profile_lit_binary_dir, config.name)

target_is_msvc = bool(re.match(r".*-windows-msvc$", config.target_triple))

if config.target_os in ["Linux"]:
    extra_link_flags = ["-ldl"]
elif target_is_msvc:
    # InstrProf is incompatible with incremental linking. Disable it as a
    # workaround.
    extra_link_flags = ["-Wl,-incremental:no"]
else:
    extra_link_flags = []

# Test suffixes.
config.suffixes = [".c", ".cpp", ".hip", ".m", ".mm", ".ll", ".test"]

# What to exclude.
config.excludes = ["Inputs"]

# Clang flags.
target_cflags = [get_required_attr(config, "target_cflags")]
clang_cflags = target_cflags + extra_link_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags

# TODO: target_cflags can sometimes contain C++ only flags like -stdlib=<FOO>, which are
#       ignored when compiling as C code. Passing this flag when compiling as C results in
#       warnings that break tests that use -Werror.
#       We remove -stdlib= from the cflags here to avoid problems, but the interaction between
#       CMake and compiler-rt's tests should be reworked so that cflags don't contain C++ only
#       flags.
clang_cflags = [
    flag.replace("-stdlib=libc++", "").replace("-stdlib=libstdc++", "")
    for flag in clang_cflags
]


def build_invocation(compile_flags, with_lto=False):
    lto_flags = []
    if with_lto and config.lto_supported:
        lto_flags += config.lto_flags
    return " " + " ".join([config.clang] + lto_flags + compile_flags) + " "


def exclude_unsupported_files_for_aix(dirname):
    for filename in os.listdir(dirname):
        source_path = os.path.join(dirname, filename)
        if os.path.isdir(source_path):
            continue
        f = open(source_path, "r")
        try:
            data = f.read()
            # rpath is not supported on AIX, exclude all tests with them.
            if ( "-rpath" in data ):
                config.excludes += [filename]
        finally:
            f.close()


# Add clang substitutions.
config.substitutions.append(("%clang ", build_invocation(clang_cflags)))
config.substitutions.append(("%clangxx ", build_invocation(clang_cxxflags)))

config.substitutions.append(
    ("%clang_profgen ", build_invocation(clang_cflags) + " -fprofile-instr-generate ")
)
config.substitutions.append(
    ("%clang_profgen=", build_invocation(clang_cflags) + " -fprofile-instr-generate=")
)
config.substitutions.append(
    (
        "%clangxx_profgen ",
        build_invocation(clang_cxxflags) + " -fprofile-instr-generate ",
    )
)
config.substitutions.append(
    (
        "%clangxx_profgen=",
        build_invocation(clang_cxxflags) + " -fprofile-instr-generate=",
    )
)

config.substitutions.append(
    ("%clang_pgogen ", build_invocation(clang_cflags) + " -fprofile-generate ")
)
config.substitutions.append(
    ("%clang_pgogen=", build_invocation(clang_cflags) + " -fprofile-generate=")
)
config.substitutions.append(
    ("%clangxx_pgogen ", build_invocation(clang_cxxflags) + " -fprofile-generate ")
)
config.substitutions.append(
    ("%clangxx_pgogen=", build_invocation(clang_cxxflags) + " -fprofile-generate=")
)

config.substitutions.append(
    ("%clang_cspgogen ", build_invocation(clang_cflags) + " -fcs-profile-generate ")
)
config.substitutions.append(
    ("%clang_cspgogen=", build_invocation(clang_cflags) + " -fcs-profile-generate=")
)
config.substitutions.append(
    ("%clangxx_cspgogen ", build_invocation(clang_cxxflags) + " -fcs-profile-generate ")
)
config.substitutions.append(
    ("%clangxx_cspgogen=", build_invocation(clang_cxxflags) + " -fcs-profile-generate=")
)

config.substitutions.append(
    ("%clang_profuse=", build_invocation(clang_cflags) + " -fprofile-instr-use=")
)
config.substitutions.append(
    ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=")
)

config.substitutions.append(
    ("%clang_pgouse=", build_invocation(clang_cflags) + " -fprofile-use=")
)
config.substitutions.append(
    ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=")
)

config.substitutions.append(
    (
        "%clang_lto_profgen=",
        build_invocation(clang_cflags, True) + " -fprofile-instr-generate=",
    )
)

if config.target_os not in [
    "Windows",
    "Darwin",
    "FreeBSD",
    "Linux",
    "NetBSD",
    "SunOS",
    "AIX",
    "Haiku",
]:
    config.unsupported = True

config.substitutions.append(
    ("%shared_lib_flag", "-dynamiclib" if (config.target_os == "Darwin") else "-shared")
)

if config.target_os in ["AIX"]:
    config.available_features.add("system-aix")
    exclude_unsupported_files_for_aix(config.test_source_root)
    exclude_unsupported_files_for_aix(config.test_source_root + "/Posix")

if config.target_arch in ["armv7l"]:
    config.unsupported = True

if config.android:
    config.unsupported = True

if config.have_curl:
    config.available_features.add("curl")

if config.target_os in ("AIX", "Darwin", "Linux"):
    config.available_features.add("continuous-mode")

# The device-profile drain (.hip) tests need:
#   - an AMD GPU exposed via the KFD kernel driver,
#   - a usable HIP install (so `clang -x hip` can build the test), and
#   - the amdgcn device profile runtime in the compiler-rt resource directory.
#
# `hip` is enabled as soon as a usable HIP install is found; `amdgpu` is
# enabled only when the device profile runtime is *also* present in the
# resource dir. The .hip drain tests gate on `amdgpu`, so a missing runtime
# (or a compiler-rt build that disabled the GPU profile runtime) leaves them
# UNSUPPORTED rather than failing -- as does a host with /dev/kfd but no ROCm
# install.
#
# Also export %hip_lib_path and %amdgpu_arch substitutions so .hip tests can
# stay portable (the existing GPU/instrprof-hip-* tests in amd-staging use
# the same names).
if os.path.exists("/dev/kfd"):
    rocm_path = os.environ.get("ROCM_PATH") or "/opt/rocm"
    hip_runtime_h = os.path.join(rocm_path, "include", "hip", "hip_runtime.h")
    hip_lib_path = os.path.join(rocm_path, "lib")
    has_hip_install = os.path.isfile(hip_runtime_h) and any(
        os.path.exists(os.path.join(hip_lib_path, n))
        for n in (
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
        )
    )
    if has_hip_install:
        config.available_features.add("hip")
        config.substitutions.append(("%hip_lib_path", hip_lib_path))
        # `native` lets clang derive the arch from the visible KFD agents,
        # which matches how the existing GPU/ tests operate when not pinned
        # to a specific gfx target.
        config.substitutions.append(("%amdgpu_arch", "native"))
        # Probe for the amdgcn device profile runtime in the resource dir.
        # `config.compiler_rt_libdir` is set by AddCompilerRT.cmake; fall back
        # to deriving it from the clang binary if that attribute is absent
        # (older compiler-rt setups).
        rt_libdir = getattr(config, "compiler_rt_libdir", None)
        if rt_libdir is None and getattr(config, "clang", None):
            clang_dir = os.path.dirname(os.path.realpath(config.clang))
            rt_libdir = os.path.join(
                os.path.dirname(clang_dir), "lib", "clang"
            )
        # The device profile runtime may be installed either under the
        # arch-suffixed name (libclang_rt.profile-amdgcn.a) or, in the newer
        # per-target resource-dir layout, the arch-less name
        # (libclang_rt.profile.a under lib/amdgcn-amd-amdhsa/). Accept either,
        # but only treat the arch-less name as the *device* runtime when it
        # lives under an amdgcn target dir so we don't match the host runtime.
        profile_rt = None
        if rt_libdir and os.path.isdir(rt_libdir):
            for root, _, files in os.walk(rt_libdir):
                if "libclang_rt.profile-amdgcn.a" in files:
                    profile_rt = os.path.join(
                        root, "libclang_rt.profile-amdgcn.a"
                    )
                    break
                if "libclang_rt.profile.a" in files and "amdgcn" in root:
                    profile_rt = os.path.join(root, "libclang_rt.profile.a")
                    break
        if profile_rt is not None:
            config.available_features.add("amdgpu")
