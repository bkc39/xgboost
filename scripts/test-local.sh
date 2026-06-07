#!/usr/bin/env bash
set -euo pipefail

# End-to-end test using the system Racket install (no Nix).
# Installs the xgboost package from candidates/, runs the unit tests and all
# CPU examples, then runs the CUDA examples (which should skip gracefully on a
# CPU-only build).

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

echo "--- clearing staged native libs ---"
rm -f xgboost/native-libs/libxgbcompat.* \
      xgboost/native-libs/libxgboost.*   \
      xgboost/native-libs/libomp.*       \
      xgboost/native-libs/libgomp.*      \
      xgboost/native-libs/libstdc++.*    \
      xgboost/native-libs/libxgbshim.*

echo "--- installing from candidates ---"
"$RACO" pkg install --name xgboost ./xgboost

echo "--- raco test xgboost/ ---"
"$RACO" test xgboost/

echo "--- examples (literate scribble/lp2; runners + checks in xgboost/examples/test/) ---"
# The CUDA harnesses (24, 25) self-skip when no GPU is available.
"$RACO" test xgboost/examples/test/

echo "--- all done ---"
