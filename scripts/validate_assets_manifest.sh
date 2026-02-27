#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/assets/manifest/assets.lock.json"
ASSET_ROOT="${REPO_ROOT}/assets"
EXPECTED_GITHUB_REPO="Jedi-Vision/spatial-audio"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "Lockfile not found: ${LOCK_FILE}" >&2
  exit 1
fi

for cmd in jq curl shasum; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} is required" >&2
    exit 1
  fi
done

release_tag="$(jq -r '.release' "${LOCK_FILE}")"
if [[ -z "${release_tag}" || "${release_tag}" == "null" ]]; then
  echo "Lockfile release tag is missing in ${LOCK_FILE}" >&2
  exit 1
fi

expected_prefix="https://github.com/${EXPECTED_GITHUB_REPO}/releases/download/${release_tag}/"
echo "Validating lockfile URLs and checksums for release ${release_tag}"

jq -c '.assets[]' "${LOCK_FILE}" | while IFS= read -r item; do
  rel_path="$(jq -r '.path' <<<"${item}")"
  url="$(jq -r '.url' <<<"${item}")"
  sha256_expected="$(jq -r '.sha256' <<<"${item}")"
  file_name="$(basename "${rel_path}")"
  expected_url="${expected_prefix}${file_name}"

  if [[ "${url}" != "${expected_url}" ]]; then
    echo "URL mismatch for ${rel_path}" >&2
    echo "  expected: ${expected_url}" >&2
    echo "  actual:   ${url}" >&2
    exit 1
  fi

  code="$(curl -I -L -s -o /dev/null -w '%{http_code}' "${url}")"
  if [[ "${code}" != "200" ]]; then
    echo "Remote asset unavailable for ${rel_path}: HTTP ${code}" >&2
    exit 1
  fi

  local_file="${ASSET_ROOT}/${rel_path}"
  if [[ -f "${local_file}" ]]; then
    sha256_actual="$(shasum -a 256 "${local_file}" | awk '{print $1}')"
    if [[ "${sha256_actual}" != "${sha256_expected}" ]]; then
      echo "Local checksum mismatch for ${rel_path}" >&2
      echo "  expected: ${sha256_expected}" >&2
      echo "  actual:   ${sha256_actual}" >&2
      exit 1
    fi
  fi

  echo "OK ${rel_path}"
done

echo "Manifest validation succeeded."
