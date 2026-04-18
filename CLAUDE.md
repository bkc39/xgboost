# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Racket bindings for XGBoost, built with Nix. Scaffolding follows the `hello-nix-raco` template: a C++ shared library (`libxgbcompat`) built with CMake that links against `pkgs.xgboost`, loaded into Racket via `ffi-lib`. Current FFI surface is a smoke-test (version + two hardcoded-data demos); real DMatrix/Booster bindings will come next.

## Build Commands

```bash
nix build                    # Build racket package (runs raco setup + raco test)
nix build .#cpp              # Build only the C++ library + run gtests
nix develop                  # Enter dev shell (auto-installs package on first entry)
./result/bin/xgboost-rkt     # Run the main module
```

Inside `nix develop`:
```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

raco test xgboost-rkt/
racket -l xgboost-rkt
```

To force re-provision of the dev-shell racket user scope: `rm -rf .racket-user/`.

## Architecture

### C++ (cpp/)

- `include/xgbcompat/xgbcompat.hpp` — C++ namespace + `extern "C"` FFI surface.
- `src/examples.cpp` — hardcoded-data regression + binary classification demos using XGBoost's C API (`XGDMatrixCreateFromMat`, `XGBoosterCreate`, `XGBoosterUpdateOneIter`, `XGBoosterPredictFromDMatrix`).
- `src/xgbcompat.cpp` — thin `extern "C"` translation layer; catches C++ exceptions and returns int rc.
- `tests/xgbcompat_test.cpp` — GoogleTest cases; covers semver-shape version, MSE-on-train sanity, probability range, and extern-C parity.
- CMake: `find_package(xgboost CONFIG REQUIRED)` pulls the nixpkgs install. Linked via `xgboost::xgboost`.

### Racket (xgboost-rkt/)

- `info.rkt` — declares `pre-install-collection` that runs the native-lib installer.
- `private/install-xgboost-native.rkt` — pre-installer; copies `libxgbcompat.*` from `$XGBOOST_RKT_NATIVE_LIB_PATH/lib` into `native-libs/` at raco-setup time.
- `private/ffi-raw.rkt` — raw bindings via `define-ffi-definer`; `define-runtime-path` points at `../native-libs` so the dylib is resolved portably.
- `private/xgboost-native.rkt` — Racket wrapper with contracts; checks rc, raises on error using `xgb-last-error`.
- `main.rkt` — public exports + `module+ main` + `module+ test`.

### Nix Flake (flake.nix)

- `packages.cpp` — cmake-built shared library + gtests.
- `packages.racket` — raco-built package; depends on cpp via `buildInputs` and env var `XGBOOST_RKT_NATIVE_LIB_PATH`.
- `packages.copy-native-libs` — helper app for non-Nix dev workflows.
- `devShells.default` — all toolchain deps + link-mode raco install.

The `--name xgboost-rkt` flag on `raco pkg install` is critical — without it raco registers under the Nix-store hash.

## Platform Notes

Primary target is darwin (aarch64-darwin and x86_64-darwin). Linux should work because the flake uses the standard shared-library extension; untested. The stdenv on darwin is clang++, which compiles C++26 out of the box.
