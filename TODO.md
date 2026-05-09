# TODO

## Phase 1: Docs And Rename

- [x] Rename the collection to `xgboost`.
- [x] Make `(require xgboost)` the high-level entrypoint.
- [x] Move contracted low-level wrappers to `(require xgboost/ffi)`.
- [x] Move direct C bindings to `(require xgboost/ffi/raw)`.
- [x] Update README, examples, and Nix commands to the new module names.

## Phase 2: High-Level API

- [x] Add `make-dmatrix`, `train`, `predict`, `save-model`, `load-model`, and eval helpers.
- [x] Accept row matrices, flat vectors, and `f32vector` feature data.
- [x] Accept labels/weights as lists, vectors, or `f32vector`s.
- [x] Coerce training parameter keys and values before calling the FFI layer.
- [x] Wrap high-level DMatrix and Booster values in Racket structs for lifetime tracking.
- [x] Add Scribble docs once the API shape settles.

## Phase 3: Missing C API Coverage

- [x] Global config, build-info, and log callback APIs.
- [x] DMatrix constructors: file/URI, CSR, CSC, dense array interface, columnar array interface.
- [x] DMatrix metadata: string feature info, uint info, info interface, slicing, save-binary, quantile cuts.
- [x] Booster APIs: reset, slice, boosted rounds, number of features.
- [x] Booster serialization/config/inspection: save/load config, dumps, attrs, feature scores.
- [x] Booster snapshots.
- [x] Custom objective training.
- [x] CPU inplace prediction APIs: dense, CSR, and columnar.
- [ ] CUDA inplace prediction APIs: CUDA array and columnar variants.
- [ ] Distributed tracker and communicator APIs. Deferred until the local API implementation is robust.

## Phase 4: CUDA And Linux Validation

- [ ] Validate Linux CPU builds in Nix.
- [ ] Add CUDA-enabled Linux/Nix target work.
- [ ] Keep macOS as CPU-only.
- [ ] Add CI coverage for Linux and Darwin.

## Phase 5: Cleanup

- [ ] Add dedicated `exn:fail:xgboost` exception type.
- [ ] Tighten DMatrix info-field and row-count validation in `xgboost/ffi`.
- [x] Track high-level DMatrix lifetimes held by high-level Booster caches.
- [ ] Track DMatrix lifetimes held by low-level `xgboost/ffi` Booster caches.
- [ ] Add ASan or Valgrind checks for the C++ shim.
