# xgboost-rkt

Racket bindings for [XGBoost](https://xgboost.readthedocs.io/), built with Nix.

Scaffolding only: the current FFI exposes the XGBoost version string and two hardcoded-data demos (regression + binary classification). It proves that libxgboost links, loads, trains, and predicts correctly from Racket.

## Quick start

```bash
nix build
./result/bin/xgboost-rkt
```

Expected output:

```
xgboost version: 3.0.5
regression first prediction: <some float>
classification first prediction: <some float in [0,1]>
```

## Development

```bash
nix develop

# build + test the C++ library standalone
cmake -S cpp -B cpp/build -G Ninja -DBUILD_TESTING=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure

# run the Racket tests
raco test xgboost-rkt/
```

## Layout

- `cpp/` — C++ wrapper library (`libxgbcompat`) built with CMake, links against `pkgs.xgboost`.
- `xgboost-rkt/` — Racket package (collection: `xgboost-rkt`), loads `libxgbcompat` via FFI.
- `flake.nix` — `cpp` and `racket` derivations.

See `CLAUDE.md` for architecture details.
