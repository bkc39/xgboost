# xgboost

Racket bindings for [XGBoost](https://xgboost.readthedocs.io/).

- **XGBoost documentation:** https://xgboost.readthedocs.io/
- **This package's API reference:** https://docs.racket-lang.org/xgboost/

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

Use `(require xgboost/foreign)` for the contracted low-level DMatrix/Booster wrappers, and `(require xgboost/foreign/raw)` for direct C FFI bindings.

## Installation

```bash
raco pkg install xgboost
```

The package ships a prebuilt native library and picks the right one for your
platform at install time. On Linux it prefers a CUDA-enabled build when one is
available and falls back to the CPU build otherwise.

## Development

Development uses [Nix](https://nixos.org/), which provides a reproducible
toolchain (Racket, CMake, the XGBoost C++ library, and the linters).

```bash
nix develop
```

Inside the dev shell, build and test the C++ library:

```bash
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

Run the Racket tests:

```bash
raco test xgboost/
racket examples/01-train-regression.rkt   # narrative example
```

`nix build` runs both the package tests and the fast assertion-backed examples
under `examples/`. Longer narrative demos stay manually runnable.

Every new public API or user-visible feature should land with an
example-backed E2E test unless that is impractical for runtime, platform, or
dependency reasons.

### Building the native library locally

`scripts/build-so.sh` builds `libxgbcompat` plus its bundled dependencies and
stages them under `xgboost/xgboost/native-libs/candidates/<platform>/`, so a plain
`raco pkg install` works afterwards without Nix. Run it from inside `nix
develop`:

```bash
./scripts/build-so.sh darwin        # macOS (CPU)         → candidates/darwin/
./scripts/build-so.sh linux         # Linux CPU           → candidates/linux-cpu/
./scripts/build-so.sh linux-cuda    # Linux CUDA (x86_64) → candidates/linux-cuda/

# Then install from the local checkout:
raco pkg install --name xgboost ./xgboost
```

### Linters

The dev shell provisions [Resyntax](https://docs.racket-lang.org/resyntax/)
(refactoring suggestions) and
[racket-review](https://pkgs.racket-lang.org/package/review) (surface-level
style/correctness checks):

```bash
resyntax analyze --directory xgboost     # report suggestions
resyntax fix --directory xgboost         # apply them in place
raco review xgboost/**/*.rkt             # surface-level lint
```

The `Nix checks` CI workflow runs `resyntax analyze` as a gate, so run
`resyntax fix` before pushing.

## Layout

- `cpp/` - C++ wrapper library (`libxgbcompat`) built with CMake, links against `pkgs.xgboost`.
- `examples/` - runnable examples; selected fast RackUnit-backed examples run by default checks.
- `xgboost/` - the multi-collection package (`collection 'multi`); install with `raco pkg install ./xgboost`.
- `xgboost/xgboost/` - the `xgboost` code collection.
- `xgboost/xgboost/main.rkt` - high-level root API for `(require xgboost)`.
- `xgboost/xgboost/foreign.rkt` - contracted low-level Racket wrappers.
- `xgboost/xgboost/foreign/raw.rkt` - direct C FFI bindings.
- `xgboost/xgboost/private/` - native library installer implementation.
- `xgboost/xgboost-doc/` - the `xgboost-doc` documentation collection (Scribble manual).
- `flake.nix` - `cpp` and `racket` derivations.

See `AGENTS.md` for architecture notes and the full set of build/test commands.
