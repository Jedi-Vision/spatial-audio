#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on macOS. Install from https://brew.sh" >&2
  exit 1
fi

brew update
brew install cmake ninja pkg-config zeromq portaudio glfw jq

if [[ -z "${VCPKG_ROOT:-}" ]]; then
  echo "VCPKG_ROOT is not set. Example:"
  echo "  export VCPKG_ROOT=\$HOME/dev/vcpkg"
fi

echo "macOS bootstrap complete."
