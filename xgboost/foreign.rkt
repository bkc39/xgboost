#lang racket/base

;; raco review does surface-level linting without macro expansion, so it
;; cannot see that every `contract-out` entry below re-exports an imported
;; identifier — it would report each as "provided but not defined".  This is
;; a pure re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the safe, contracted FFI layer.  `(require xgboost/foreign)`
;; exposes the full low-level surface; all implementation lives in
;; foreign/*.  Handles are returned as `dmatrix` / `booster` wrapper structs
;; whose underlying C handles are GC-reclaimed.
;;
;; `(require (submod xgboost/foreign unsafe))` additionally exposes the
;; explicit-free helpers `dmatrix-free!` / `booster-free!` — see the `unsafe`
;; submodule below.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require (except-in ffi/unsafe ->)
         ffi/vector
         racket/contract
         "foreign/structs.rkt"
         "foreign/global.rkt"
         "foreign/array-interface.rkt"
         "foreign/dmatrix/create.rkt"
         "foreign/dmatrix/metadata.rkt"
         "foreign/dmatrix/ops.rkt"
         "foreign/booster/core.rkt"
         "foreign/booster/predict.rkt"
         "foreign/booster/persist.rkt"
         "foreign/booster/inspect.rkt")

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [xgboost-build-info (-> string?)]
  [xgboost-get-global-config (-> string?)]
  [xgboost-set-global-config! (-> string? void?)]
  [xgboost-register-log-callback! (-> (-> string? any/c) void?)]
  [dmatrix? (-> any/c boolean?)]
  [dmatrix-create-from-mat
   (->* (f32vector? exact-nonnegative-integer? exact-nonnegative-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-uri (-> string? dmatrix?)]
  [dmatrix-create-from-dense-array-interface
   (->* (string?) (string?) dmatrix?)]
  [dmatrix-create-from-csr-array-interface
   (->* (string? string? string? exact-nonnegative-integer?)
        (string?)
        dmatrix?)]
  [dmatrix-create-from-csc-array-interface
   (->* (string? string? string? exact-nonnegative-integer?)
        (string?)
        dmatrix?)]
  [dmatrix-create-from-columnar-array-interface
   (->* (string?) (string?) dmatrix?)]
  [dmatrix-create-from-dense
   (->* (f32vector? exact-positive-integer? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-csr
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-csc
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-columnar
   (->* ((listof f32vector?))
        (real?)
        dmatrix?)]
  [dmatrix-slice
   (->* (dmatrix? (or/c s32vector?
                         (listof exact-nonnegative-integer?)
                         (vectorof exact-nonnegative-integer?)))
        (#:allow-groups? any/c)
        dmatrix?)]
  [dmatrix-set-float-info! (-> dmatrix? string? f32vector? void?)]
  [dmatrix-set-uint-info! (-> dmatrix? string? u32vector? void?)]
  [dmatrix-set-info-from-interface! (-> dmatrix? string? string? void?)]
  [dmatrix-set-feature-info! (-> dmatrix? string? (listof string?) void?)]
  [dmatrix-nrow (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-ncol (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-get-float-info (-> dmatrix? string? f32vector?)]
  [dmatrix-get-uint-info (-> dmatrix? string? u32vector?)]
  [dmatrix-get-feature-info (-> dmatrix? string? (listof string?))]
  [dmatrix-num-non-missing (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix->list (-> dmatrix? (listof (listof real?)))]
  [dmatrix-save-binary! (->* (dmatrix? path-string?) (#:silent? any/c) void?)]
  [dmatrix-get-quantile-cut (->* (dmatrix?) (string?) (values string? string?))]
  [array-interface->u64vector (-> string? u64vector?)]
  [array-interface->f32vector (-> string? f32vector?)]
  [dmatrix-show (->* (dmatrix?) (output-port?) void?)]
  [booster? (-> any/c boolean?)]
  [booster-create (->* () ((listof dmatrix?)) booster?)]
  [booster-reset! (-> booster? void?)]
  [booster-slice
   (->* (booster? exact-integer? exact-integer?)
        (exact-positive-integer?)
        booster?)]
  [booster-boosted-rounds (-> booster? exact-nonnegative-integer?)]
  [booster-num-feature (-> booster? exact-nonnegative-integer?)]
  [booster-set-param! (-> booster? string? string? void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-train-one-iter! (-> booster? exact-integer? dmatrix? f32vector? f32vector? void?)]
  [booster-predict
   (->* (booster? dmatrix?)
        (#:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?))
        f32vector?)]
  [booster-predict-from-dense
   (->* (booster? f32vector? exact-positive-integer? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-predict-from-csr
   (->* (booster? u64vector? u32vector? f32vector? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-predict-from-columnar
   (->* (booster? (listof f32vector?))
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-save-model! (-> booster? path-string? void?)]
  [booster-load-model! (-> booster? path-string? void?)]
  [booster-save-model-to-bytes
   (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [booster-load-model-from-bytes! (-> booster? bytes? void?)]
  [booster-serialize-to-bytes (-> booster? bytes?)]
  [booster-unserialize-from-bytes! (-> booster? bytes? void?)]
  [booster-save-json-config (-> booster? string?)]
  [booster-load-json-config! (-> booster? string? void?)]
  [booster-set-attr! (-> booster? string? string? void?)]
  [booster-delete-attr! (-> booster? string? void?)]
  [booster-get-attr (-> booster? string? (or/c #f string?))]
  [booster-get-attr-names (-> booster? (listof string?))]
  [booster-set-feature-info! (-> booster? string? (listof string?) void?)]
  [booster-get-feature-info (-> booster? string? (listof string?))]
  [booster-dump-model
   (->* (booster?)
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c)
        (listof string?))]
  [booster-dump-model-with-features
   (->* (booster? (listof string?) (listof string?))
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c)
        (listof string?))]
  [booster-feature-score
   (->* (booster?)
        (#:importance-type string?
         #:feature-names (or/c #f (listof string?))
         #:config (or/c #f string?))
        hash?)]
  [booster-eval-one-iter
   (-> booster? exact-integer?
       (listof (cons/c string? dmatrix?))
       string?)]
  [parse-eval-line (-> string? (hash/c string? real?))]
  [dmatrix-handle (-> dmatrix? cpointer?)]
  [dmatrix-rows (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-cols (-> dmatrix? exact-nonnegative-integer?)]
  [booster-handle (-> booster? cpointer?)]
  [booster-cache (-> booster? (listof dmatrix?))]))

;; Explicit-free escape hatch.  The high-level `xgboost` API and the safe
;; `xgboost/foreign` surface above rely on the GC: the raw layer registers a
;; finalizer at allocation time, so handles are reclaimed automatically once
;; unreachable.  These providers are for callers who need to release a handle
;; on a deterministic schedule (long-lived processes, large in-flight
;; DMatrices) or who are migrating from a legacy explicit-free codebase.
;; Both are idempotent: they flip the cpointer tag so a second call raises an
;; `exn:fail:contract` instead of double-freeing at the C level.
(module+ unsafe
  (require racket/contract
           (only-in "foreign/structs.rkt"
                    dmatrix? booster? dmatrix-free! booster-free!))
  (provide (contract-out
            [dmatrix-free! (-> dmatrix? void?)]
            [booster-free! (-> booster? void?)])))
