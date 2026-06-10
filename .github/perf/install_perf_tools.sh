#!/usr/bin/env bash
# Install kernel-matched perf inside the job container (no host package changes).
#
# manylinux (el8) cannot run Ubuntu-built perf directly (glibc mismatch). We:
#   1) download linux-tools-$(uname -r) Ubuntu debs into the workspace via docker
#   2) extract them to ${PERF_CHROOT_ROOT}
#   3) run perf through chroot + bind-mounted /proc,/sys,/dev (needs --privileged)
set -euo pipefail

kernel="$(uname -r)"
kshort="$(echo "${kernel}" | cut -d- -f1-2)"
azure_tools_ver="${kernel%-azure}"
perf_wrapper="/usr/local/bin/perf"

# Job files live under GITHUB_WORKSPACE. Docker bind mounts must use the host path
# (/home/runner/_work/...); /__w/... bind sources are empty on the host daemon.
resolve_docker_bind_path() {
  local path="$1"
  case "${path}" in
    /__w/*) echo "/home/runner/_work/${path#/__w/}" ;;
    *) echo "${path}" ;;
  esac
}

job_workspace="${GITHUB_WORKSPACE:-.}"
perf_root="${job_workspace}/.kernel-perf-tools"
docker_perf_root="$(resolve_docker_bind_path "${perf_root}")"
chroot_root="${perf_root}/root"
debs_dir="${perf_root}/debs"
export PERF_CHROOT_ROOT="${chroot_root}"
export PERF_KERNEL="${kernel}"

echo "perf install: kernel=${kernel} workspace=${job_workspace} docker_mount=${docker_perf_root}"

ensure_docker_cli() {
  if command -v docker >/dev/null 2>&1; then
    return 0
  fi
  if [[ ! -S /var/run/docker.sock ]]; then
    return 1
  fi
  local version="24.0.9"
  local tmp
  tmp="$(mktemp -d)"
  echo "Fetching static docker CLI ${version}..."
  curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${version}.tgz" \
    | tar -xzf - -C "${tmp}" docker/docker
  install -m 755 "${tmp}/docker/docker" /usr/local/bin/docker
  export PATH="/usr/local/bin:${PATH}"
}

extract_debs_into_root() {
  shopt -s nullglob
  local deb
  local docker_debs docker_root
  docker_debs="$(resolve_docker_bind_path "${debs_dir}")"
  docker_root="$(resolve_docker_bind_path "${chroot_root}")"
  for deb in "${debs_dir}"/*.deb; do
    dpkg-deb -x "${deb}" "${chroot_root}" 2>/dev/null \
      || docker run --rm -v "${docker_debs}:/debs:ro" -v "${docker_root}:/root" \
        ubuntu:22.04 bash -ce "dpkg-deb -x '/debs/$(basename "${deb}")' /root"
  done
}

download_linux_tools_debs() {
  if [[ ! -S /var/run/docker.sock ]]; then
    echo "Note: /var/run/docker.sock not available for linux-tools download." >&2
    return 1
  fi
  ensure_docker_cli || return 1

  rm -rf "${perf_root}"
  mkdir -p "${debs_dir}" "${chroot_root}"
  # Ensure host-side mount target exists for nested docker.
  mkdir -p "$(dirname "${docker_perf_root}")" "${docker_perf_root}/debs" "${docker_perf_root}/root" 2>/dev/null || true

  echo "Downloading linux-tools-${kernel} debs into ${perf_root} (container-only install)..."

  if ! docker run --rm \
    -v "${docker_perf_root}:/out" \
    ubuntu:22.04 bash -ce "
      set -eux
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      mkdir -p /out/debs /out/root

      try_download() {
        apt-get install -y -d --no-install-recommends \"\$@\"
      }

      if try_download \
          'linux-tools-${kernel}' \
          'linux-cloud-tools-${kernel}' \
          linux-tools-common \
          bash \
          dash; then
        :
      elif try_download 'linux-tools-${kshort}' linux-tools-common; then
        :
      else
        try_download linux-tools-common linux-tools-generic
      fi

      # apt -d skips packages already present in ubuntu:22.04 (notably libc6/ld-linux).
      # perf is dynamically linked and needs the interpreter inside the chroot rootfs.
      apt-get install -y -d --reinstall libc6 libgcc-s1 libstdc++6 \
        || apt-get download libc6 libgcc-s1 libstdc++6 \
        || true

      shopt -s nullglob
      for deb in /var/cache/apt/archives/*.deb; do
        cp -f \"\${deb}\" /out/debs/
        dpkg-deb -x \"\${deb}\" /out/root
      done
      ls -la /out/debs/
    "; then
    return 1
  fi

  shopt -s nullglob
  local debs=("${debs_dir}"/*.deb)
  if (( ${#debs[@]} == 0 )); then
    echo "ERROR: no .deb packages downloaded to ${debs_dir}" >&2
    echo "Hint: expected debs at ${debs_dir} (docker mount ${docker_perf_root})" >&2
    return 1
  fi
  echo "Downloaded ${#debs[@]} debs to ${debs_dir}"

  if [[ ! -d "${chroot_root}/usr" ]]; then
    extract_debs_into_root
  fi
  return 0
}

find_chroot_perf() {
  [[ -d "${chroot_root}/usr/lib" ]] || return 1
  local candidate

  is_chroot_elf() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    # Skip #!/bin/sh wrappers from linux-tools-common; they need a shell in the chroot.
    [[ "$(head -c 1 "${path}" 2>/dev/null || true)" == "#" ]] && return 1
    file "${path}" 2>/dev/null | grep -q 'ELF' || return 1
    return 0
  }

  # Prefer the kernel-matched ELF from linux-azure-tools, not the wrapper script.
  for candidate in \
    "${chroot_root}/usr/lib/linux-azure-tools-${azure_tools_ver}/perf" \
    "${chroot_root}/usr/lib/linux-tools-${kernel}/perf" \
    "${chroot_root}/usr/lib/linux-tools-${kshort}/perf" \
    "${chroot_root}/usr/lib/linux-tools/${kernel}/perf" \
    "${chroot_root}/usr/lib/linux-tools/${kshort}/perf"; do
    if is_chroot_elf "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] || continue
    if is_chroot_elf "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done < <(find "${chroot_root}/usr/lib" -name perf -type f 2>/dev/null)

  return 1
}

setup_chroot_mounts() {
  local d
  for d in proc sys dev; do
    mkdir -p "${chroot_root}/${d}"
    if mountpoint -q "${chroot_root}/${d}" 2>/dev/null; then
      continue
    fi
    mount --bind "/${d}" "${chroot_root}/${d}"
  done
}

verify_chroot_runtime() {
  local ld_linux="${chroot_root}/lib64/ld-linux-x86-64.so.2"
  if [[ ! -e "${ld_linux}" ]]; then
    ld_linux="${chroot_root}/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
  fi
  if [[ ! -e "${ld_linux}" ]]; then
    echo "ERROR: chroot missing dynamic linker (libc6 not extracted?)" >&2
    find "${chroot_root}" -name 'ld-linux-x86-64.so.2' 2>/dev/null | head -5 >&2 || true
    return 1
  fi
  echo "Chroot dynamic linker: ${ld_linux#"${chroot_root}"}"
}

install_perf_wrapper() {
  local chroot_perf="$1"
  local relpath="${chroot_perf#"${chroot_root}"}"
  mkdir -p /usr/local/bin
  cat >"${perf_wrapper}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec chroot "${chroot_root}" "${relpath}" "\$@"
EOF
  chmod 755 "${perf_wrapper}"
  export PATH="/usr/local/bin:${PATH}"
  echo "Installed perf wrapper -> chroot ${relpath}"
  echo "Selected chroot perf: ${chroot_perf} ($(file -b "${chroot_perf}" 2>/dev/null || echo unknown))"
}

chroot_perf_path=""
if chroot_perf_path="$(find_chroot_perf 2>/dev/null || true)" && [[ -n "${chroot_perf_path}" ]]; then
  echo "Reusing existing chroot perf: ${chroot_perf_path}"
elif download_linux_tools_debs; then
  chroot_perf_path="$(find_chroot_perf || true)"
fi

if [[ -z "${chroot_perf_path}" || ! -e "${chroot_perf_path}" ]]; then
  echo "ERROR: no kernel-matched perf for ${kernel}." >&2
  echo "Downloaded debs:" >&2
  ls -la "${debs_dir}" 2>&1 | head -10 >&2 || true
  echo "Chroot usr/lib/linux-tools:" >&2
  ls -la "${chroot_root}/usr/lib/linux-tools" 2>&1 | head -10 >&2 || true
  find "${chroot_root}/usr/lib" -maxdepth 3 -name 'perf*' 2>&1 | head -10 >&2 || true
  exit 127
fi

setup_chroot_mounts
verify_chroot_runtime
install_perf_wrapper "${chroot_perf_path}"

echo 1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null \
  || echo "WARNING: could not set perf_event_paranoid (non-fatal)"

# Diagnostics for logs: kernel PMU / hw event visibility (do not fail the job).
echo "=== perf diagnostics: hardware events (perf list hw) ===" >&2
perf list hw 2>&1 || true
echo "=== perf diagnostics: system-wide 1s sample (perf stat -a -- sleep 1) ===" >&2
perf stat -a -- sleep 1 2>&1 || true
echo "=== end perf diagnostics ===" >&2

perf_check="$(
  perf stat -x \; -e cycles:u,instructions:u,task-clock,duration_time -- true 2>&1
)" || {
  echo "ERROR: perf stat check failed" >&2
  echo "${perf_check}" | head -20 >&2
  exit 1
}

if grep -q '<not supported>' <<<"${perf_check}"; then
  echo "ERROR: perf hardware counters (cycles:u/instructions:u) not supported" >&2
  echo "perf: $(command -v perf)  kernel: ${kernel}" >&2
  echo "${perf_check}" >&2
  exit 1
fi

echo "${perf_check}" | head -6
