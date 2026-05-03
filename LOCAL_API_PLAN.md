# Local API Implementation Plan

This plan tracks the non-CUDA, non-distributed path toward a robust local
Racket XGBoost binding. Use it as the working reference for upcoming changes.

## Scope

- Focus on local CPU workflows: data construction, metadata, training,
  prediction, model inspection, serialization, and serving-oriented inference.
- Keep macOS CPU-only.
- Hold distributed tracker and communicator APIs out of scope until the local
  implementation is substantially complete.
- CUDA APIs are tracked separately and should not block local CPU completeness.
- Every new public API should include a runnable example with RackUnit checks
  unless impractical.

## Phase 1: DMatrix Metadata And Dataset Operations

Goal: make datasets fully describable, inspectable, sliceable, and reusable.

- String feature info:
  - `dmatrix-set-feature-info!`
  - `dmatrix-get-feature-info`
  - fields: `"feature_name"` and `"feature_type"`
- Integer metadata:
  - `dmatrix-set-uint-info!`
  - `dmatrix-get-uint-info`
  - fields such as `"group"`, `"qid"`, and other XGBoost-supported uint info.
- General info interface:
  - expose `XGDMatrixSetInfoFromInterface` for array-interface-backed metadata.
  - use this where XGBoost has deprecated older typed setters.
- Dataset operations:
  - row slicing via `XGDMatrixSliceDMatrixEx`
  - binary DMatrix save via `XGDMatrixSaveBinary`
  - binary reload through the existing URI constructor where supported.
- Quantile cuts:
  - expose `XGDMatrixGetQuantileCut`
  - return copied JSON array-interface descriptors for indptr and data.
- Examples:
  - `examples/14-dmatrix-metadata.rkt`
  - `examples/15-dmatrix-slicing-binary.rkt`
  - `examples/16-quantile-cuts.rkt`

## Phase 2: Booster Lifecycle And Inspection

Goal: make trained models inspectable and manageable from Racket.

- Lifecycle:
  - reset training caches
  - slice boosted rounds
  - report boosted rounds
  - report number of features
- Configuration:
  - save JSON config
  - load JSON config
- Attributes:
  - set/get one attribute
  - list attribute names
- Dumps and feature inspection:
  - dump model text/json
  - dump with feature names/types
  - feature score / importance APIs
- Examples:
  - booster config round-trip
  - attrs and dumps
  - feature importance from named features

## Phase 3: CPU Inplace Prediction

Goal: support serving-style prediction without forcing users to build a
DMatrix first.

- Dense CPU array-interface prediction.
- CSR CPU array-interface prediction.
- Columnar CPU array-interface prediction.
- Shared prediction-shape handling where useful.
- Examples:
  - dense inplace prediction equals DMatrix prediction
  - CSR and columnar variants on small data

## Phase 4: Callback-Based Local Extensions

Goal: support custom local training behavior without distributed complexity.

- Custom objective update support.
- Custom metric/eval support if the C API and callback lifetime story are
  acceptable from Racket.
- Carefully document callback lifetimes and exception behavior.

## Phase 5: Robustness And Cleanup

Goal: make failures clearer and handle ownership rigorously.

- Dedicated `exn:fail:xgboost` exception type.
- Validate known DMatrix info fields and metadata lengths where possible.
- Track DMatrix lifetimes held by low-level `xgboost/ffi` Booster caches.
- Add ASan or Valgrind checks for the native shim.
- Expand docs after each API group settles.

## Deferred

- Distributed tracker and communicator APIs.
- CUDA array / CUDA columnar APIs.
- Linux CUDA packaging.
