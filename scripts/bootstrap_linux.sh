#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This bootstrap script currently supports apt-based Linux distributions." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  git \
  curl \
  jq \
  libzmq3-dev \
  portaudio19-dev \
  libglfw3-dev

required_cmake_version="4.1.0"
detected_cmake_version="$(cmake --version | awk 'NR==1 { print $3 }')"

if [[ "$(printf '%s\n' "${required_cmake_version}" "${detected_cmake_version}" | sort -V | head -n1)" != "${required_cmake_version}" ]]; then
  echo "CMake >= ${required_cmake_version} is required. Detected: ${detected_cmake_version}" >&2
  echo "Install a newer CMake (for example from Kitware) and rerun this script." >&2
  exit 1
fi

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Example:"
  echo "  export VCPKG_ROOT=\$HOME/dev/vcpkg"
fi

echo "Linux bootstrap complete."
