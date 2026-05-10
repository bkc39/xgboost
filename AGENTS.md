# AGENTS.md

## Project Overview

This repository provides Racket bindings for XGBoost. The Racket package is now the `xgboost` collection:

- `(require xgboost)` is the high-level API for ordinary Racket data. **Use this by default.** DMatrix and Booster are wrapper structs whose underlying handles are reclaimed by Racket's GC; user code never calls `*-free!`.
- `(require xgboost/ffi)` is the contracted low-level wrapper layer. Returns raw cpointers tagged `_DMatrix` / `_Booster`. The raw layer wires `(allocator … (deallocator))` so handles are still GC-reclaimed; the safe surface no longer exports explicit-free helpers.
- `(require (submod xgboost/ffi unsafe))` exposes `dmatrix-free!` and `booster-free!` for callers that need deterministic release (long-lived processes, very large in-flight matrices, migration from legacy explicit-free code). Both are idempotent: a second call hits the cpointer tag guard and raises `exn:fail:contract` instead of double-freeing.
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
raco test \
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
  examples/23-custom-objective.rkt \
  examples/26-booster-snapshot.rkt
racket -l xgboost
```

`scripts/build-so.sh` builds the native library and stages it under
`xgboost/native-libs/candidates/<platform>/`. After running it, a plain
`raco pkg install` works without Nix:

```bash
./scripts/build-so.sh linux          # → candidates/linux-cpu/
./scripts/build-so.sh linux-cuda     # → candidates/linux-cuda/  (x86_64 only)
./scripts/build-so.sh darwin         # → candidates/darwin/

# Then install the Racket package (pre-installer picks the right candidate):
raco pkg install --name xgboost ./xgboost
```

On Linux the pre-installer prefers a CUDA candidate if present, then falls back
to the CPU one.

The default `nix build` runs both `raco test ./xgboost/` and
the selected fast assertion-backed examples under `examples/`. Longer
narrative demos should stay manually runnable, but they are not all part of
the default package check.

To force re-provisioning of the dev-shell Racket user scope, remove `.racket-user/`.

## End-to-End Testing

### Nix path (CI-equivalent)

```bash
nix build          # builds cpp, installs pkg, runs raco test + examples
```

### raco catalog simulation (pre-submission check)

This reproduces what the Racket package catalog does: fresh install from
candidates, no Nix environment, no XGBOOST_NATIVE_LIB_PATH.

```bash
# Start from a clean slate inside nix develop (provides racket)
nix develop --command bash -c "
  raco pkg remove xgboost 2>/dev/null || true
  rm -f xgboost/native-libs/libxgbcompat.* \
        xgboost/native-libs/libxgboost.*   \
        xgboost/native-libs/libgomp.so.1
  raco pkg install --name xgboost ./xgboost
  raco test xgboost/
  raco test \
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
    examples/23-custom-objective.rkt \
    examples/26-booster-snapshot.rkt
"
```

The pre-installer reads from `native-libs/candidates/<platform>/` and copies
the `.so` files into `native-libs/` — this is the path that `raco pkg install`
from the catalog would follow. Tests must pass with no Nix store paths on
`LD_LIBRARY_PATH`.

### CUDA examples (requires physical NVIDIA GPU)

```bash
nix develop .#cuda
racket examples/24-cuda-regression.rkt
racket examples/25-cuda-classification.rkt
```

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
- Selected files in `examples/` are assertion-backed examples; each exports
  `run-example`, prints concise output from `module+ main`, and verifies
  behavior from `module+ test`.

Every new public API or user-visible feature should include an
example-backed E2E test in the same change set unless that is impractical.

## CUDA Build (x86_64-linux only)

```bash
# Build CUDA-enabled .so and install to xgboost/native-libs/
./scripts/build-so.sh linux-cuda
# or equivalently:
nix build .#cpp-cuda
nix run .#copy-native-libs-cuda

# Enter the CUDA dev shell
nix develop .#cuda

# Run CUDA examples (requires a physical NVIDIA GPU)
racket examples/24-cuda-regression.rkt
racket examples/25-cuda-classification.rkt
```

CUDA examples check availability via `xgboost-build-info` (JSON key `USE_CUDA`) and skip
gracefully on CPU-only builds. They are NOT in the default `nix build` test suite.

## Nix Notes

`flake.nix` provides:

- `packages.cpp` - CMake-built shared library plus gtests.
- `packages.racket` - Racket package build and tests.
- `packages.copy-native-libs` - helper for non-Nix workflows.
- `devShells.default` - toolchain and link-mode Racket package install.
- `packages.cpp-cuda` - CUDA-enabled shared library (x86_64-linux only).
- `packages.copy-native-libs-cuda` - copies CUDA `.so` to `native-libs/` (x86_64-linux only).
- `devShells.cuda` - CUDA toolchain and link-mode install (x86_64-linux only).

The package install command should use `--name xgboost` so Racket registers the intended collection name instead of a store-derived name.

## Roadmap

Current local implementation phases are tracked in `LOCAL_API_PLAN.md`.
`TODO.md` remains a compact checklist for broad project status.
