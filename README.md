# xgboost

Racket bindings for [XGBoost](https://xgboost.readthedocs.io/), built with Nix.

The default API is high level:

```racket
#lang racket

(require xgboost)

(define dtrain
  (make-dmatrix '((1.0 2.0 0.5)
                  (2.0 1.0 1.5)
                  (3.0 0.5 0.0)
                  (0.5 3.0 2.0))
                #:labels '(3.5 3.5 6.5 2.0)))

(define booster
  (train dtrain
         #:objective "reg:squarederror"
         #:max-depth 2
         #:eta 0.2
         #:verbosity 0
         #:rounds 10))

(predict booster dtrain)
```

Use `(require xgboost/ffi)` for the contracted low-level DMatrix/Booster wrappers, and `(require xgboost/ffi/raw)` for direct C FFI bindings.

## Quick Start

```bash
nix build
./result/bin/xgboost
```

## Development

```bash
nix develop

# build + test the C++ library standalone
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

# run the Racket tests
raco test xgboost/
raco test examples/11-global-apis.rkt \
  examples/12-dmatrix-constructors.rkt \
  examples/13-high-level-root-api.rkt

# run a narrative example manually
racket examples/01-train-regression.rkt
```

`nix build` runs both the package tests and the fast assertion-backed examples
under `examples/`. Longer narrative demos stay manually runnable, but they are
not all part of the default package check.

Every new public API or user-visible feature should land with an
example-backed E2E test unless that is impractical for runtime, platform, or
dependency reasons.

## Layout

- `cpp/` - C++ wrapper library (`libxgbcompat`) built with CMake, links against `pkgs.xgboost`.
- `examples/` - runnable examples; selected fast RackUnit-backed examples run by default checks.
- `xgboost/main.rkt` - high-level root API for `(require xgboost)`.
- `xgboost/ffi.rkt` - contracted low-level Racket wrappers.
- `xgboost/ffi/raw.rkt` - direct C FFI bindings.
- `xgboost/private/` - native library installer implementation.
- `flake.nix` - `cpp` and `racket` derivations.

See `AGENTS.md` for architecture notes and the current roadmap.
