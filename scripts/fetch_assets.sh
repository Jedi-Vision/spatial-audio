#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/assets/manifest/assets.lock.json"
ASSET_ROOT="${REPO_ROOT}/assets"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "Lockfile not found: ${LOCK_FILE}" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse ${LOCK_FILE}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download assets" >&2
  exit 1
fi

if ! command -v shasum >/dev/null 2>&1; then
  echo "shasum is required to verify checksums" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Fetching assets into ${ASSET_ROOT}"

jq -c '.assets[]' "${LOCK_FILE}" | while IFS= read -r item; do
  rel_path="$(jq -r '.path' <<<"${item}")"
  url="$(jq -r '.url' <<<"${item}")"
  sha256_expected="$(jq -r '.sha256' <<<"${item}")"

  dest="${ASSET_ROOT}/${rel_path}"
  dest_dir="$(dirname "${dest}")"
  mkdir -p "${dest_dir}"

  tmp_file="${tmp_dir}/$(basename "${rel_path}")"
  echo "  - ${rel_path}"
  curl --fail --location --silent --show-error "${url}" --output "${tmp_file}"

  sha256_actual="$(shasum -a 256 "${tmp_file}" | awk '{print $1}')"
  if [[ "${sha256_actual}" != "${sha256_expected}" ]]; then
    echo "Checksum mismatch for ${rel_path}" >&2
    echo "  expected: ${sha256_expected}" >&2
    echo "  actual:   ${sha256_actual}" >&2
    exit 1
  fi

  mv "${tmp_file}" "${dest}"
done

echo "Assets fetched and verified."
