#!/usr/bin/env bash
set -euo pipefail

# Fails if active repository text still contains pbme/PBME.
# Allow migration notes/changelog files if they ever exist.
if rg -n "pbme|PBME" \
  --glob '!diabatic_matrices.json' \
  --glob '!CHANGELOG*' \
  --glob '!MIGRATION*' \
  --glob '!*migration*' \
  --glob '!check_no_pbme_refs.sh'
then
  echo "Found disallowed pbme/PBME references." >&2
  exit 1
fi

echo "No disallowed pbme/PBME references found."
