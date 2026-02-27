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

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Example:"
  echo "  export VCPKG_ROOT=\$HOME/dev/vcpkg"
fi

echo "Linux bootstrap complete."
