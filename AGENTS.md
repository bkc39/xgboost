# AGENTS.md

## Project Overview

This repository provides Racket bindings for XGBoost. The Racket package is now the `xgboost` collection:

- `(require xgboost)` is the high-level API for ordinary Racket data.
- `(require xgboost/ffi)` is the contracted low-level wrapper layer.
- `(require xgboost/ffi/raw)` is the direct C FFI layer.

The native bridge is a C++ shared library, `libxgbcompat`, built with CMake and linked against `pkgs.xgboost`.

## Build Commands

```bash
nix build
nix build .#cpp
nix develop
./result/bin/xgboost
```

Inside `nix develop`:

```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

raco test xgboost/
racket -l xgboost
```

To force re-provisioning of the dev-shell Racket user scope, remove `.racket-user/`.

## Architecture

### C++

- `cpp/include/xgbcompat/xgbcompat.hpp` - namespace plus `extern "C"` FFI surface.
- `cpp/src/examples.cpp` - hardcoded regression and binary classification demos.
- `cpp/src/xgbcompat.cpp` - C translation layer; catches C++ exceptions and returns integer status codes.
- `cpp/tests/xgbcompat_test.cpp` - GoogleTest coverage for native behavior.

### Racket

- `xgboost/info.rkt` - package metadata and native-library pre-install hook.
- `xgboost/main.rkt` - high-level API.
- `xgboost/ffi.rkt` - safe contracted low-level wrappers.
- `xgboost/ffi/raw.rkt` - direct `define-ffi-definer` bindings.
- `xgboost/private/install-xgboost-native.rkt` - copies `libxgbcompat.*` from `$XGBOOST_NATIVE_LIB_PATH/lib` into `native-libs/`.

## Nix Notes

`flake.nix` provides:

- `packages.cpp` - CMake-built shared library plus gtests.
- `packages.racket` - Racket package build and tests.
- `packages.copy-native-libs` - helper for non-Nix workflows.
- `devShells.default` - toolchain and link-mode Racket package install.

The package install command should use `--name xgboost` so Racket registers the intended collection name instead of a store-derived name.

## Roadmap

Current phases are tracked in `TODO.md`: docs/rename, high-level API, missing C API coverage, CUDA/Linux validation, and cleanup.
