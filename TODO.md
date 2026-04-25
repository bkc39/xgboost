# TODO

Running list of deferred work. Roughly ordered by dependency; section
headers group related items so it stays scannable as it grows.

## Higher-level package (future)

The current `xgboost-rkt` package surfaces the raw FFI bindings directly.
Future work: a higher-level Racket API that wraps these and hides the
"bring your own `f32vector`, know XGBoost's info-field vocabulary" surface
area. The raw bindings stay as an `xgboost-rkt/unsafe` or `xgboost-rkt/raw`
entry point; the high-level API becomes the default `(require xgboost-rkt)`
experience.

Working name: `xgboost-rkt` (high-level) on top of `xgboost-rkt/unsafe` (raw).

## Contract tightening — DMatrix

Gotchas currently documented in prose / caught only at the C boundary. Each
should become an enforced contract (or wrapper-level check) on the raw API,
with clearer errors than `XGDMatrix*: ...`.

- [ ] **Info-field whitelist.** `dmatrix-set-float-info!` currently accepts
      any string and lets XGBoost reject it. Define a `dmatrix-float-field/c`
      contract matching XGBoost's float-valued info fields
      (`"label"`, `"weight"`, `"base_margin"`, `"label_lower_bound"`,
      `"label_upper_bound"`, `"feature_weights"`). Raise a contract error up
      front instead of surfacing `XGDMatrixSetFloatInfo: ...`.
- [ ] **Label/weight length must equal nrow.** Most float-info fields must
      have length = number of rows in the DMatrix. We currently pass through
      any length and let XGBoost complain. Teach the wrapper to track `nrow`
      on the handle (via a parallel hash table keyed on cpointer, or a small
      Racket struct that wraps the cpointer) and add a `(= (f32vector-length
      vals) (dmatrix-nrow dm))` contract.
- [ ] **Parallel `set-uint-info!` / `set-str-info!`.** Some info fields
      (`"group"`, `"qid"`) are uint, not float. Not yet bound. When added,
      each gets its own whitelist + length contract in the same style.
- [ ] **Missing-value coercion.** `missing` is typed `real?` but XGBoost
      reads it as `float`. Document the float-round-trip behavior (e.g.
      `1e40` becomes `+inf.f`) or narrow the contract to
      `single-flonum?`-representable reals.
- [ ] **Shape helpers.** Users often have `(list-of (list-of real?))` or
      `vector?`-of-`vector?` data, not a flat row-major `f32vector`. Add
      `rows->f32vector` / `matrix->dmatrix` helpers in the high-level API.
      The high-level `make-dmatrix` should take rows + labels + optional
      weights in one call instead of three raw steps.

## Contract tightening — Booster (when added)

Not yet implemented (Phase 2). Capture the known gotchas now so the
wrapper ships with contracts from day one.

- [ ] **Parameter whitelist + type coercion.** `XGBoosterSetParam` takes
      string key + string value; wrong parameter names are silently
      ignored by XGBoost in some versions. Enforce a known-params table and
      auto-stringify Racket values (numbers → `number->string`, booleans →
      `"true"`/`"false"`).
- [ ] **Objective compatibility.** `"binary:logistic"` requires
      labels in {0,1}; `"reg:squarederror"` does not. Validate labels
      against objective at train time.
- [ ] **Prediction output copy.** `XGBoosterPredictFromDMatrix` returns a
      booster-owned `const float*` — the memory is invalidated by the next
      prediction call. The high-level API must copy into Racket-owned
      storage (`f32vector-copy` into a fresh vector) before returning.
      Design the FFI to do the copy in the C shim so Racket never sees the
      borrowed pointer.
- [ ] **DMatrix lifetime vs Booster cache.** `XGBoosterCreate` takes an
      array of DMatrix handles as its training cache; those DMatrices must
      outlive the Booster. Track the reference in the Racket wrapper (hold
      the DMatrix cpointers in a box on the Booster struct) so GC doesn't
      reclaim the training data out from under a live Booster.

## Error-surface polish

- [ ] `xgb-last-error` is thread-local in C. If we ever call the FFI from
      Racket `place`s, `future`s, or OS threads we don't control, error
      messages will vanish. Document the constraint in `SKILL.md` (already
      noted) and consider a mutex-protected global buffer if/when we go
      multi-threaded.
- [ ] Wrapper-raised exceptions currently use plain `error`. Consider a
      dedicated `exn:fail:xgboost` struct so downstream callers can
      discriminate C errors from contract failures.

## Testing / CI

- [ ] Valgrind or AddressSanitizer job on the C++ side. The current
      `getrusage`-based gtest is a coarse smoke test; ASan would catch
      genuine leaks and use-after-free deterministically. Add a
      `packages.cpp-asan` flake output and wire it into `nix flake check`.
- [ ] Racket-side finalizer-path observability. Right now the
      "finalizer path: GC reclaims DMatrices" test only verifies "does not
      crash." If Racket ever exposes finalizer-fired counters we should use
      them; until then, the C-side RSS test is the backstop.
- [ ] Cross-platform CI (at least Linux x86_64 in addition to darwin).
      Mentioned in CLAUDE.md as "untested"; lock it in so the
      `ru_maxrss`-units branch stays honest.

## Documentation

- [ ] Scribble docs for the public API. `info.rkt` declares
      `(define scribblings '())`; replace with a real doc tree once the
      high-level API is stable enough to be worth documenting.
- [ ] Usage examples in `README.md` (doesn't exist yet). Include the REPL
      session from the conversation that prompted this TODO (DMatrix
      create + set-label + free) plus, later, a full train/predict walkthrough.
