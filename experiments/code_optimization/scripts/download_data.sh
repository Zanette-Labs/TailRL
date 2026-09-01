#!/usr/bin/env bash
# Fetch the merged PIE test-case corpus: one directory per problem holding
# input.<i>.txt / output.<i>.txt for every case the reward grades against.
#
# This is the PIE paper's own test-case release (~100 cases per problem, the
# AlphaCode-generated set), not the 3-cases-per-problem public subset. The reward's
# correctness gate runs a rollout against every usable case for its problem, so a
# thinner corpus silently changes what "correct" means.
#
#   92 MB download, ~760 MB extracted across 688,315 files. Give it a filesystem
#   that does not mind small files; on a networked one, extraction dominates.
#
# Idempotent: re-running with the corpus already present verifies and exits.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

GDRIVE_ID="1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr"
SHA256="02f21225ba963ec9de89b3a5e42f2e9647d8333b717a02cf5bd8b4d088b8309c"
DEST="${PIE_TEST_CASE_DIR}"
STAGE="$(dirname "${DEST}")"
TARBALL="${STAGE}/merged_test_cases.tar.gz"

if [ -d "${DEST}" ] && [ -f "${DEST}/version.txt" ]; then
  n=$(find "${DEST}" -maxdepth 1 -mindepth 1 -type d | wc -l)
  echo "[data] test cases already present: ${DEST} (${n} problems)"
  echo "[data] export PIE_TEST_CASE_DIR=${DEST}"
  exit 0
fi

mkdir -p "${STAGE}"

if [ ! -f "${TARBALL}" ]; then
  command -v gdown >/dev/null 2>&1 || {
    echo "FATAL: gdown is not installed (pip install gdown). The corpus is hosted on" >&2
    echo "       Google Drive, which needs gdown's confirm-token dance for large files." >&2
    exit 2
  }
  echo "[data] downloading the merged test-case corpus (92 MB)..."
  gdown --id "${GDRIVE_ID}" -O "${TARBALL}"
fi

echo "[data] verifying checksum..."
got="$(sha256sum "${TARBALL}" | cut -d' ' -f1)"
if [ "${got}" != "${SHA256}" ]; then
  echo "FATAL: checksum mismatch for ${TARBALL}" >&2
  echo "       expected ${SHA256}" >&2
  echo "       got      ${got}" >&2
  echo "       Delete it and re-run; a partial or Drive-quota-page download is the usual cause." >&2
  exit 2
fi

echo "[data] extracting (688k small files, this takes a few minutes)..."
tmp="${STAGE}/.extract.$$"
mkdir -p "${tmp}"
tar xzf "${TARBALL}" -C "${tmp}"
# The archive holds a single top-level directory; move whatever it is into place.
inner="$(find "${tmp}" -maxdepth 1 -mindepth 1 -type d | head -1)"
mv "${inner}" "${DEST}"
rmdir "${tmp}" 2>/dev/null || true

n=$(find "${DEST}" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[data] done: ${n} problems under ${DEST}"
echo "[data] the tarball is no longer needed: rm ${TARBALL}"
echo
echo "  export PIE_TEST_CASE_DIR=${DEST}"
