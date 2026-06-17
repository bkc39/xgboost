# AGENTS.md

## Project Overview

This repository provides Racket bindings for XGBoost. The Racket package is now the `xgboost` collection:

- `(require xgboost)` is the high-level API for ordinary Racket data. **Use this by default.** DMatrix and Booster are wrapper structs whose underlying handles are reclaimed by Racket's GC; user code never calls `*-free!`.
- `(require xgboost/foreign)` is the contracted low-level wrapper layer. Returns `dmatrix` / `booster` wrapper structs over cpointers tagged `_DMatrix` / `_Booster`. The raw layer wires `(allocator … (deallocator))` so handles are still GC-reclaimed; the safe surface no longer exports explicit-free helpers.
- `(require (submod xgboost/foreign unsafe))` exposes `dmatrix-free!` and `booster-free!` for callers that need deterministic release (long-lived processes, very large in-flight matrices, migration from legacy explicit-free code). Both are idempotent: a second call hits the cpointer tag guard and raises `exn:fail:contract` instead of double-freeing.
- `(require xgboost/foreign/raw)` is the direct C FFI layer.

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

The dev shell also provisions two Racket linters into the user scope —
[Resyntax](https://docs.racket-lang.org/resyntax/) (refactoring suggestions)
and [racket-review](https://pkgs.racket-lang.org/package/review) (surface-level
style/correctness checks):

```bash
resyntax analyze --directory xgboost          # report suggestions
resyntax fix --directory xgboost              # apply them in place
raco review xgboost/**/*.rkt                  # surface-level lint
```

Note: `raco review` does not expand macros, so the pure re-export facades
(`main.rkt`, `foreign.rkt`, `foreign/raw.rkt`) and `info.rkt` carry a
`#|review: ignore|#` directive — every `contract-out` re-export would
otherwise be misreported as "provided but not defined".

The `Nix checks` workflow runs `resyntax analyze` as a CI gate (the
`resyntax` job): a pull request fails if Resyntax reports any suggestion, so
run `resyntax fix` before pushing.

`scripts/build-so.sh` builds the native library and stages it under
`xgboost/xgboost/native-libs/candidates/<platform>/`. After running it, a plain
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
  rm -f xgboost/xgboost/native-libs/libxgbcompat.* \
        xgboost/xgboost/native-libs/libxgboost.*   \
        xgboost/xgboost/native-libs/libgomp.so.1
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

The repository ships a single multi-collection package rooted at `xgboost/`
(`xgboost/info.rkt` declares `collection 'multi` plus package metadata and the
native-library pre-install hook). It provides two collections:

- `xgboost/xgboost/` - the `xgboost` code collection (require path `xgboost`).
- `xgboost/xgboost-doc/` - the `xgboost-doc` documentation collection
  (`xgboost.scrbl`); rendered output lands in `xgboost/xgboost-doc/doc/`.

`raco pkg install ./xgboost` installs both. `(require xgboost)` and
`(require xgboost/foreign[/raw])` are unchanged. The code collection is split
into small, logically coherent modules; every top-level module path is a thin
*re-export facade*, with the implementation in sub-collection modules
(target ≤ 500 lines/file).

- `xgboost/xgboost/info.rkt` - collection-level info (test timeouts).
- `xgboost/xgboost/main.rkt` - facade for the high-level API; re-exports `core/*`.
- `xgboost/xgboost/core/*.rkt` - high-level implementation: `coerce`, `global`,
  `dmatrix`, `booster`, `train`, `predict`, `persist`.
- `xgboost/xgboost/foreign.rkt` - facade for the safe contracted layer; re-exports
  `foreign/*` and defines the `unsafe` submodule.
- `xgboost/xgboost/foreign/*.rkt` - safe-layer implementation: `error`, `structs`,
  `global`, `array-interface`, `dmatrix/{create,metadata,ops}`,
  `booster/{core,predict,persist,inspect}`.
- `xgboost/xgboost/foreign/raw.rkt` - facade for the direct C FFI layer; re-exports
  `foreign/raw/{library,global,dmatrix,booster}.rkt` (`define-ffi-definer`
  bindings).
- `xgboost/xgboost/tests/*.rkt` - cross-cutting integration tests; module-local unit
  tests live in `module+ test` submodules.
- `xgboost/xgboost/private/install-xgboost-native.rkt` - copies `libxgbcompat.*` from `$XGBOOST_NATIVE_LIB_PATH/lib` into `native-libs/`.
- Selected files in `examples/` are assertion-backed examples; each exports
  `run-example`, prints concise output from `module+ main`, and verifies
  behavior from `module+ test`.

Every new public API or user-visible feature should include an
example-backed E2E test in the same change set unless that is impractical.

## CUDA Build (x86_64-linux only)

```bash
# Build CUDA-enabled .so and install to xgboost/xgboost/native-libs/
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
