#!/usr/bin/env bash
set -euo pipefail

# Installs perf_launcher.sh into a directory prepended to PATH for compiler launches.
# Default destination matches Jenkins perf nightly (/opt/rocm/bin); Multi-Arch CI passes
# "${BUILD_DIR}/bin" instead.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest_dir="${1:-$(readlink -f /opt/rocm)/bin}"

mkdir -p "${dest_dir}"
install -m 755 "${script_dir}/perf_launcher.sh" "${dest_dir}/perf_launcher.sh"
echo "Installed ${dest_dir}/perf_launcher.sh"
