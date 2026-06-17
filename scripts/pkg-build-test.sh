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
# The literate examples now ship inside the package (xgboost/examples/), so the
# `--package xgboost` run below also exercises xgboost/examples/test/. Before the
# tests we render the manual from a package-only copy of xgboost/ — the same view
# the catalog builds (source=...?path=xgboost) — so a doc @lp-include that escapes
# the package root fails here instead of on the catalog.

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
rm -f xgboost/native-libs/libxgbcompat.* \
      xgboost/native-libs/libxgboost.*   \
      xgboost/native-libs/libomp.*       \
      xgboost/native-libs/libgomp.*      \
      xgboost/native-libs/libstdc++.*    \
      xgboost/native-libs/libxgbshim.*

echo "--- installing the collection from candidates (no Nix, no env var) ---"
unset XGBOOST_NATIVE_LIB_PATH || true
# --auto installs declared deps (e.g. polars) non-interactively; the catalog
# build host resolves them the same way.
"$RACO" pkg install --auto --name xgboost ./xgboost

echo "--- doc-build guard: render the manual from a package-only copy (mirrors the catalog) ---"
# The catalog builds source=...?path=xgboost, i.e. only the xgboost/ subtree.
# Render from an isolated copy of just xgboost/ so any out-of-package @lp-include
# (e.g. ../../examples/...) fails here exactly as it does on pkg-build. Rendering
# needs only label-phase bindings, so PLTCOLLECTS resolves `xgboost` to the copy
# without the native library.
guard_root=$(mktemp -d)
trap 'rm -rf "$guard_root"' EXIT
cp -R xgboost "$guard_root/xgboost"
find "$guard_root/xgboost" -name compiled -type d -exec rm -rf {} + 2>/dev/null || true
PLTCOLLECTS="$guard_root:" "$RACO" scribble --dest "$guard_root/doc" \
  "$guard_root/xgboost/scribblings/xgboost.scrbl"

echo "--- raco test --drdr --package xgboost (the pkg-build test step) ---"
# -j 2 mirrors a modest builder; --drdr supplies the 90s timeout + check-stderr.
"$RACO" test -j 2 --drdr --package xgboost
