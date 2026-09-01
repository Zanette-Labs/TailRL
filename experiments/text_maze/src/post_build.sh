#!/bin/bash
# Backward-compatible dataset post-processing entrypoint.
# Prefer scripts/prepare_dataset.sh for new runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${1:-data/main_1.3M.jsonl}"
exec bash scripts/prepare_dataset.sh "$JSONL" data/sft_data data/main_parquet
