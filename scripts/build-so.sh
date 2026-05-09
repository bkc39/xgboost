#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"

usage() {
  echo "Usage: $0 <target>"
  echo "  targets: linux | linux-cuda | darwin"
  exit 1
}

case "$TARGET" in
  linux)
    echo "Building CPU-only libxgbcompat for Linux..."
    nix build .#cpp --print-build-logs
    mkdir -p xgboost/native-libs/candidates/linux-cpu
    cp -v --no-preserve=mode result/lib/libxgbcompat.* xgboost/native-libs/candidates/linux-cpu/
    echo "Done. CPU .so installed to xgboost/native-libs/candidates/linux-cpu/"
    ;;

  linux-cuda)
    SYSTEM="$(uname -m)-linux"
    if [ "$SYSTEM" != "x86_64-linux" ]; then
      echo "Error: linux-cuda target requires x86_64-linux (got $SYSTEM)" >&2
      exit 1
    fi
    echo "Building CUDA-enabled libxgbcompat for Linux (x86_64)..."
    nix build .#cpp-cuda --print-build-logs
    mkdir -p xgboost/native-libs/candidates/linux-cuda
    cp -v --no-preserve=mode result/lib/libxgbcompat.* xgboost/native-libs/candidates/linux-cuda/
    echo "Done. CUDA .so installed to xgboost/native-libs/candidates/linux-cuda/"
    ;;

  darwin)
    if [ "$(uname)" != "Darwin" ]; then
      echo "Error: darwin target requires macOS" >&2
      exit 1
    fi
    echo "Building CPU-only libxgbcompat for macOS..."
    nix build .#cpp --print-build-logs
    mkdir -p xgboost/native-libs/candidates/darwin
    cp -v --no-preserve=mode result/lib/libxgbcompat.* xgboost/native-libs/candidates/darwin/
    echo "Done. CPU .dylib installed to xgboost/native-libs/candidates/darwin/"
    ;;

  *)
    usage
    ;;
esac
