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
find xgboost examples -name "compiled" -type d -exec rm -rf {} + 2>/dev/null || true

echo "--- removing previous xgboost install ---"
"$RACO" pkg remove xgboost 2>/dev/null || true

echo "--- clearing staged native libs ---"
rm -f xgboost/native-libs/libxgbcompat.* \
      xgboost/native-libs/libxgboost.*   \
      xgboost/native-libs/libomp.*       \
      xgboost/native-libs/libgomp.*

echo "--- installing from candidates ---"
"$RACO" pkg install --name xgboost ./xgboost

echo "--- raco test xgboost/ ---"
"$RACO" test xgboost/

echo "--- CPU examples ---"
"$RACO" test \
  examples/11-global-apis.rkt \
  examples/12-dmatrix-constructors.rkt \
  examples/13-high-level-root-api.rkt \
  examples/14-dmatrix-metadata.rkt \
  examples/15-dmatrix-slicing-binary.rkt \
  examples/16-quantile-cuts.rkt \
  examples/17-booster-lifecycle-config.rkt \
  examples/18-booster-attrs.rkt \
  examples/19-booster-dumps-feature-scores.rkt \
  examples/20-inplace-predict-dense.rkt \
  examples/21-inplace-predict-csr.rkt \
  examples/22-inplace-predict-columnar.rkt \
  examples/23-custom-objective.rkt

echo "--- CUDA examples (expect graceful skip on CPU build) ---"
"$RACKET" examples/24-cuda-regression.rkt
"$RACKET" examples/25-cuda-classification.rkt

echo "--- all done ---"
