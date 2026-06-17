#!/usr/bin/env bash
set -euo pipefail

# Reproduce the pkg-build.racket-lang.org test step exactly.
#
# The package server installs only the `xgboost` collection (path=xgboost) and
# then runs:
#
#   raco test --drdr --package xgboost
#
# `--drdr` (see compiler/commands/test.rkt) sets a 90s per-test timeout AND
# `check-stderr?`, which treats *any* write to stderr as a test failure. Our
# other CI runs plain `raco test xgboost/`, which ignores stderr — that is why
# the catalog can fail while our regular checks stay green. This script closes
# that gap: run it to catch stderr/timeout regressions before the catalog does.
#
# It deliberately does NOT run the examples/ tests (those live at the repo root
# and are not part of the published package, so pkg-build never sees them).

RACKET=$(command -v racket 2>/dev/null || true)
RACO=$(command -v raco 2>/dev/null || true)
if [ -z "$RACKET" ] || [ -z "$RACO" ]; then
  echo "Error: racket/raco not found on PATH" >&2
  exit 1
fi
echo "Using $("$RACKET" --version)"

cd "$(dirname "$0")/.."

echo "--- cleaning compiled bytecode ---"
find xgboost -name "compiled" -type d -exec rm -rf {} + 2>/dev/null || true

echo "--- removing previous xgboost install ---"
"$RACO" pkg remove xgboost 2>/dev/null || true

echo "--- clearing staged native libs (force a clean candidate install) ---"
rm -f xgboost/xgboost/native-libs/libxgbcompat.* \
      xgboost/xgboost/native-libs/libxgboost.*   \
      xgboost/xgboost/native-libs/libomp.*       \
      xgboost/xgboost/native-libs/libgomp.*      \
      xgboost/xgboost/native-libs/libstdc++.*    \
      xgboost/xgboost/native-libs/libxgbshim.*

echo "--- installing the collection from candidates (no Nix, no env var) ---"
unset XGBOOST_NATIVE_LIB_PATH || true
"$RACO" pkg install --name xgboost ./xgboost

echo "--- raco test --drdr --package xgboost (the pkg-build test step) ---"
# -j 2 mirrors a modest builder; --drdr supplies the 90s timeout + check-stderr.
"$RACO" test -j 2 --drdr --package xgboost
