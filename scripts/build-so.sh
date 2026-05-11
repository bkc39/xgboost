#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"

usage() {
  echo "Usage: $0 <target>"
  echo "  targets: linux | linux-cuda | linux-aarch64 | darwin"
  exit 1
}

# Bundle libxgboost, libgomp, and libstdc++ alongside libxgbcompat so
# raco pkg install works without Nix.  Sets RPATH=$ORIGIN on libxgbcompat.so
# and libxgboost.so so the dynamic loader resolves libgomp/libstdc++ from
# the install directory.  Bundling libstdc++.so.6 is needed because the
# Racket package build container ships an older libstdc++ that may lack
# GLIBCXX symbols required by the Nix-built libxgboost.so (>=3.4.31).
bundle_linux() {
  local dest="$1" flake_target="$2" target_arch="${3:-x86_64}"
  local xgboost_so="" libgomp_so="" libstdcxx_so=""
  while IFS= read -r p; do
    [ -z "$xgboost_so" ]   && [ -f "$p/lib/libxgboost.so" ]   && xgboost_so="$p/lib/libxgboost.so"
    [ -z "$libgomp_so" ]   && [ -f "$p/lib/libgomp.so.1" ]    && libgomp_so="$p/lib/libgomp.so.1"
    [ -z "$libstdcxx_so" ] && [ -f "$p/lib/libstdc++.so.6" ]  && libstdcxx_so="$p/lib/libstdc++.so.6"
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

  if [ -n "$libstdcxx_so" ]; then
    cp -vL --no-preserve=mode "$libstdcxx_so" "$dest/libstdc++.so.6"
    "$patchelf" --remove-rpath "$dest/libstdc++.so.6"
    echo "Bundled libstdc++.so.6 (Nix RPATH stripped)"
  else
    echo "Warning: could not locate libstdc++.so.6 in build closure" >&2
  fi

  "$patchelf" --set-rpath '$ORIGIN' "$dest/libxgbcompat.so"
  echo "Set RPATH=\$ORIGIN on libxgbcompat.so"

  # polyfill-glibc only fully supports __isoc23_* downgrade on x86_64 today
  # (the aarch64 backend segfaults on libxgboost.so).  Skip the step on
  # aarch64: that candidate ships at GLIBC_2.38, which is fine because the
  # only aarch64 surface we ship to is Ubuntu 24.04+ (glibc 2.39+).
  if [ "$target_arch" = "x86_64" ]; then
    polyfill_glibc_linux "$dest"
  else
    echo "Skipping glibc polyfill on $target_arch (binary keeps GLIBC_2.38 dep)"
  fi
}

# Rewrite the bundled ELF binaries so they only require glibc symbols
# available on the target version (Ubuntu 22.04 / pkg-build.racket-lang.org).
# Without this, the Nix toolchain leaves references to GLIBC_2.38 symbols
# (e.g. __isoc23_strtol) that don't exist on older systems.  polyfill-glibc
# statically links small shims into the binaries to satisfy those references.
# 2.35 is the lowest target reachable today; lowering further would require
# resolving __libc_single_threaded@GLIBC_2.32.
polyfill_glibc_linux() {
  local dest="$1"
  local polyfill
  polyfill=$(nix build --no-link --print-out-paths .#polyfill-glibc 2>/dev/null)/bin/polyfill-glibc
  if [ ! -x "$polyfill" ]; then
    echo "Warning: polyfill-glibc not available, skipping glibc downgrade" >&2
    return
  fi
  echo "Polyfilling bundled libs to require only glibc <= 2.35..."
  for f in libxgboost.so libxgbcompat.so libgomp.so.1 libstdc++.so.6; do
    if [ -f "$dest/$f" ]; then
      "$polyfill" --target-glibc=2.35 "$dest/$f"
      echo "  $f -> max glibc dep now: $(objdump -T "$dest/$f" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -V -u | tail -1)"
    fi
  done
}

bundle_darwin() {
  local dest="$1"
  local xgboost_dylib="" libomp_dylib=""
  while IFS= read -r p; do
    [ -z "$xgboost_dylib" ] && [ -f "$p/lib/libxgboost.dylib" ] && xgboost_dylib="$p/lib/libxgboost.dylib"
    [ -z "$libomp_dylib" ]  && [ -f "$p/lib/libomp.dylib" ]    && libomp_dylib="$p/lib/libomp.dylib"
  done < <(nix path-info -r ".#cpp" 2>/dev/null)

  # Patch libxgbcompat.dylib: fix install name, rewrite Nix dep on libxgboost, add rpath.
  chmod +w "$dest/libxgbcompat.dylib"
  install_name_tool -id @rpath/libxgbcompat.dylib "$dest/libxgbcompat.dylib"
  local xgboost_ref
  xgboost_ref=$(otool -L "$dest/libxgbcompat.dylib" | awk '/libxgboost/{print $1}')
  if [ -n "$xgboost_ref" ]; then
    install_name_tool -change "$xgboost_ref" @rpath/libxgboost.dylib "$dest/libxgbcompat.dylib"
  fi
  install_name_tool -add_rpath @loader_path/. "$dest/libxgbcompat.dylib"
  echo "Patched libxgbcompat.dylib: install name, libxgboost dep, and rpath"

  # Bundle and fix libxgboost.dylib.
  if [ -n "$xgboost_dylib" ]; then
    cp -v "$xgboost_dylib" "$dest/"
    chmod +w "$dest/libxgboost.dylib"
    install_name_tool -id @rpath/libxgboost.dylib "$dest/libxgboost.dylib"
    # Strip all non-@ rpaths (Nix store paths, stray /lib, etc.)
    for rp in $(otool -l "$dest/libxgboost.dylib" | grep '^ *path ' | awk '{print $2}' | grep -v '^@'); do
      install_name_tool -delete_rpath "$rp" "$dest/libxgboost.dylib"
    done
    install_name_tool -add_rpath @loader_path/. "$dest/libxgboost.dylib"
    echo "Bundled libxgboost.dylib with @loader_path/. rpath"
  else
    echo "Warning: could not locate libxgboost.dylib in build closure" >&2
  fi

  # Bundle and fix libomp.dylib.
  if [ -n "$libomp_dylib" ]; then
    cp -v "$libomp_dylib" "$dest/"
    chmod +w "$dest/libomp.dylib"
    install_name_tool -id @rpath/libomp.dylib "$dest/libomp.dylib"
    for rp in $(otool -l "$dest/libomp.dylib" | grep '^ *path ' | awk '{print $2}' | grep -v '^@'); do
      install_name_tool -delete_rpath "$rp" "$dest/libomp.dylib"
    done
    install_name_tool -add_rpath @loader_path/. "$dest/libomp.dylib"
    echo "Bundled libomp.dylib with @loader_path/. rpath"
  else
    echo "Warning: could not locate libomp.dylib in build closure" >&2
  fi
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
    # The CUDA libxgboost.so is ~140 MB, exceeding GitHub's 100 MB per-file
    # limit, so the unpacked dir stays gitignored.  Ship a reproducible
    # gzip tarball instead; the pre-installer extracts it at raco install.
    echo "Packing linux-cuda candidate into reproducible tarball..."
    tar --sort=name \
        --mtime='2026-05-10 22:00:00 UTC' \
        --owner=0 --group=0 --numeric-owner \
        -czf xgboost/native-libs/candidates/linux-cuda.tar.gz \
        -C xgboost/native-libs/candidates linux-cuda
    echo "Tarball: $(ls -lh xgboost/native-libs/candidates/linux-cuda.tar.gz | awk '{print $5}')"
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
    bundle_linux xgboost/native-libs/candidates/linux-aarch64 cpp-aarch64 aarch64
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
    cp -v result/lib/libxgbcompat.* xgboost/native-libs/candidates/darwin/
    bundle_darwin xgboost/native-libs/candidates/darwin
    echo "Done. CPU .dylib installed to xgboost/native-libs/candidates/darwin/"
    ;;

  *)
    usage
    ;;
esac
