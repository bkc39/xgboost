#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"

usage() {
  echo "Usage: $0 <target>"
  echo "  targets: linux | linux-cuda | linux-aarch64 | darwin"
  exit 1
}

# Bundle libxgboost and libgomp alongside libxgbcompat so raco pkg install
# works without Nix.  Sets RPATH=$ORIGIN on both libxgbcompat.so and
# libxgboost.so; strips the Nix-store RUNPATH from libgomp.so.1 so its
# remaining deps (libpthread, libm, libc) resolve via normal system paths.
bundle_linux() {
  local dest="$1" flake_target="$2"
  local xgboost_so="" libgomp_so=""
  while IFS= read -r p; do
    [ -z "$xgboost_so" ] && [ -f "$p/lib/libxgboost.so" ] && xgboost_so="$p/lib/libxgboost.so"
    [ -z "$libgomp_so" ] && [ -f "$p/lib/libgomp.so.1" ]  && libgomp_so="$p/lib/libgomp.so.1"
  done < <(nix path-info -r ".#${flake_target}" 2>/dev/null)

  local patchelf
  patchelf=$(nix build --no-link --print-out-paths nixpkgs#patchelf 2>/dev/null)/bin/patchelf

  if [ -n "$xgboost_so" ]; then
    cp -v --no-preserve=mode "$xgboost_so" "$dest/"
    "$patchelf" --set-rpath '$ORIGIN' "$dest/libxgboost.so"
    echo "Bundled libxgboost.so with RPATH=\$ORIGIN"
  else
    echo "Warning: could not locate libxgboost.so in build closure" >&2
  fi

  if [ -n "$libgomp_so" ]; then
    cp -v --no-preserve=mode "$libgomp_so" "$dest/"
    "$patchelf" --remove-rpath "$dest/libgomp.so.1"
    echo "Bundled libgomp.so.1 (Nix RPATH stripped)"
  else
    echo "Warning: could not locate libgomp.so.1 in build closure" >&2
  fi

  "$patchelf" --set-rpath '$ORIGIN' "$dest/libxgbcompat.so"
  echo "Set RPATH=\$ORIGIN on libxgbcompat.so"
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

  linux-aarch64)
    SYSTEM="$(uname -m)-linux"
    if [ "$SYSTEM" != "x86_64-linux" ]; then
      echo "Error: linux-aarch64 cross-build requires an x86_64-linux host (got $SYSTEM)" >&2
      exit 1
    fi
    echo "Cross-compiling CPU-only libxgbcompat for aarch64-linux..."
    nix build .#cpp-aarch64 --print-build-logs
    mkdir -p xgboost/native-libs/candidates/linux-aarch64
    cp -v --no-preserve=mode result/lib/libxgbcompat.* \
      xgboost/native-libs/candidates/linux-aarch64/
    bundle_linux xgboost/native-libs/candidates/linux-aarch64 cpp-aarch64
    echo "Done. aarch64 .so installed to xgboost/native-libs/candidates/linux-aarch64/"
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
