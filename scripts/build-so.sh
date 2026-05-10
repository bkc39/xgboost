#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"

usage() {
  echo "Usage: $0 <target>"
  echo "  targets: linux | linux-cuda | darwin"
  exit 1
}

# Bundle libxgboost alongside libxgbcompat so that raco pkg install works
# without Nix on the target machine.  Sets RPATH=$ORIGIN so the loader
# finds libxgboost.so next to libxgbcompat.so in the same directory.
bundle_linux() {
  local dest="$1" flake_target="$2"
  local xgboost_so=""
  while IFS= read -r p; do
    if [ -f "$p/lib/libxgboost.so" ]; then
      xgboost_so="$p/lib/libxgboost.so"
      break
    fi
  done < <(nix path-info -r ".#${flake_target}" 2>/dev/null)

  if [ -z "$xgboost_so" ]; then
    echo "Warning: could not locate libxgboost.so in build closure — skipping bundle" >&2
    return
  fi
  cp -v --no-preserve=mode "$xgboost_so" "$dest/"
  local patchelf
  patchelf=$(nix build --no-link --print-out-paths nixpkgs#patchelf 2>/dev/null)/bin/patchelf
  "$patchelf" --set-rpath '$ORIGIN' "$dest/libxgbcompat.so"
  echo "Bundled $(basename "$xgboost_so") and set RPATH=\$ORIGIN on libxgbcompat.so"
}

bundle_darwin() {
  local dest="$1"
  local xgboost_dylib=""
  while IFS= read -r p; do
    if [ -f "$p/lib/libxgboost.dylib" ]; then
      xgboost_dylib="$p/lib/libxgboost.dylib"
      break
    fi
  done < <(nix path-info -r ".#cpp" 2>/dev/null)

  if [ -z "$xgboost_dylib" ]; then
    echo "Warning: could not locate libxgboost.dylib in build closure — skipping bundle" >&2
    return
  fi
  cp -v --no-preserve=mode "$xgboost_dylib" "$dest/"
  install_name_tool -add_rpath @loader_path/. "$dest/libxgbcompat.dylib"
  echo "Bundled $(basename "$xgboost_dylib") and set rpath on libxgbcompat.dylib"
}

case "$TARGET" in
  linux)
    echo "Building CPU-only libxgbcompat for Linux..."
    nix build .#cpp --print-build-logs
    mkdir -p xgboost/native-libs/candidates/linux-cpu
    cp -v --no-preserve=mode result/lib/libxgbcompat.* xgboost/native-libs/candidates/linux-cpu/
    bundle_linux xgboost/native-libs/candidates/linux-cpu cpp
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
    bundle_linux xgboost/native-libs/candidates/linux-cuda cpp-cuda
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
    bundle_darwin xgboost/native-libs/candidates/darwin
    echo "Done. CPU .dylib installed to xgboost/native-libs/candidates/darwin/"
    ;;

  *)
    usage
    ;;
esac
