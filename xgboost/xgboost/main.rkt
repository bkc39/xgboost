#lang racket/base

;; raco review does surface-level linting without macro expansion, so it
;; cannot see that every `contract-out` entry below re-exports an imported
;; identifier — it would report each as "provided but not defined".  This is
;; a pure re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the high-level XGBoost API.  `(require xgboost)` exposes the
;; ergonomic surface for ordinary Racket data; all implementation lives in
;; core/*.  DMatrix and Booster are the `dmatrix` / `booster` wrapper structs
;; from the foreign layer — their C handles are GC-reclaimed, so user code
;; never frees them.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require ffi/vector
         racket/contract
         ;; Wrapper-struct predicates and accessors come straight from the
         ;; foreign layer; re-exported so users see one consistent surface
         ;; regardless of which entry point they require.
         (only-in "foreign.rkt"
                  dmatrix? dmatrix-rows dmatrix-cols
                  booster? booster-cache)
         "core/global.rkt"
         "core/dmatrix.rkt"
         "core/booster.rkt"
         "core/predict.rkt"
         "core/train.rkt"
         "core/persist.rkt")

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [xgboost-build-info (-> string?)]
  [xgboost-get-global-config (-> string?)]
  [xgboost-set-global-config! (-> string? void?)]
  [xgboost-register-log-callback! (-> (-> string? any/c) void?)]
  [make-dmatrix
   (->* (any/c)
        (#:nrow (or/c #f exact-positive-integer?)
         #:ncol (or/c #f exact-positive-integer?)
         #:missing real?
         #:labels (or/c #f any/c)
         #:weights (or/c #f any/c))
        dmatrix?)]
  [make-dmatrix-from-csr
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [make-dmatrix-from-csc
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [make-dmatrix-from-columnar
   (->* ((listof f32vector?))
        (real?)
        dmatrix?)]
  [make-dmatrix-from-uri
   (->* ((or/c path-string? string?))
        (#:format (or/c #f "libsvm" "csv")
         #:silent? any/c)
        dmatrix?)]
  [dmatrix-rows (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-cols (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix->list (-> dmatrix? (listof (listof real?)))]
  [dmatrix-show (->* (dmatrix?) (output-port?) void?)]
  [dmatrix-slice
   (->* (dmatrix? (or/c list? vector? s32vector?))
        (#:allow-groups? any/c)
        dmatrix?)]
  [dmatrix-save-binary!
   (->* (dmatrix? path-string?) (#:silent? any/c) void?)]
  [dmatrix-set-label! (-> dmatrix? any/c void?)]
  [dmatrix-set-weight! (-> dmatrix? any/c void?)]
  [dmatrix-set-base-margin! (-> dmatrix? any/c void?)]
  [dmatrix-set-label-lower-bound! (-> dmatrix? any/c void?)]
  [dmatrix-set-label-upper-bound! (-> dmatrix? any/c void?)]
  [dmatrix-set-group! (-> dmatrix? any/c void?)]
  [dmatrix-set-feature-names! (-> dmatrix? (listof string?) void?)]
  [dmatrix-set-feature-types! (-> dmatrix? (listof string?) void?)]
  [dmatrix-label (-> dmatrix? (listof real?))]
  [dmatrix-weight (-> dmatrix? (listof real?))]
  [dmatrix-base-margin (-> dmatrix? (listof real?))]
  [dmatrix-group-ptr (-> dmatrix? (listof exact-nonnegative-integer?))]
  [dmatrix-feature-names (-> dmatrix? (listof string?))]
  [dmatrix-feature-types (-> dmatrix? (listof string?))]
  [dmatrix-quantile-cut
   (-> dmatrix? (values (listof exact-nonnegative-integer?) f32vector?))]
  [booster-num-feature (-> booster? exact-nonnegative-integer?)]
  [booster-boosted-rounds (-> booster? exact-nonnegative-integer?)]
  [booster-reset! (-> booster? void?)]
  [booster-slice
   (->* (booster? exact-integer? exact-integer?)
        (exact-positive-integer?)
        booster?)]
  [booster-set-attr! (-> booster? string? string? void?)]
  [booster-attr (-> booster? string? (or/c #f string?))]
  [booster-attr-names (-> booster? (listof string?))]
  [booster-delete-attr! (-> booster? string? void?)]
  [booster-set-feature-names! (-> booster? (listof string?) void?)]
  [booster-set-feature-types! (-> booster? (listof string?) void?)]
  [booster-feature-names (-> booster? (listof string?))]
  [booster-feature-types (-> booster? (listof string?))]
  [booster-config (-> booster? string?)]
  [booster-set-config! (-> booster? string? void?)]
  [booster-dump
   (->* (booster?)
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c
         #:feature-names (or/c #f (listof string?))
         #:feature-types (or/c #f (listof string?)))
        (listof string?))]
  [booster-feature-score
   (->* (booster?)
        (#:importance-type string?
         #:feature-names (or/c #f (listof string?))
         #:config (or/c #f string?))
        (hash/c symbol? any/c))]
  [predict-from-dense
   (->* (booster? any/c)
        (#:nrow (or/c #f exact-positive-integer?)
         #:ncol (or/c #f exact-positive-integer?)
         #:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [predict-from-csr
   (->* (booster? u64vector? u32vector? f32vector? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [predict-from-columnar
   (->* (booster? (listof f32vector?))
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [train
   (->* (dmatrix?)
        (#:params any/c
         #:rounds exact-nonnegative-integer?
         #:evals (listof (cons/c string? dmatrix?))
         #:objective (or/c #f any/c)
         #:objective-fn (or/c #f (-> f32vector? dmatrix? any))
         #:eta (or/c #f any/c)
         #:max-depth (or/c #f any/c)
         #:num-class (or/c #f any/c)
         #:eval-metric (or/c #f any/c)
         #:verbosity (or/c #f any/c))
        booster?)]
  [booster-set-param! (-> booster? any/c any/c void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-train-one-iter!
   (-> booster? exact-integer? dmatrix? any/c any/c void?)]
  [booster->bytes (-> booster? bytes?)]
  [bytes->booster (-> bytes? booster?)]
  [predict
   (->* (booster? dmatrix?)
        (#:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [save-model (-> booster? path-string? void?)]
  [load-model (-> path-string? booster?)]
  [make-booster (-> booster?)]
  [save-model-to-bytes (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [load-model-from-bytes (-> bytes? booster?)]
  [eval-one-iter (-> booster? exact-integer?
                     (listof (cons/c string? dmatrix?))
                     string?)]
  [booster-cache (-> booster? (listof dmatrix?))]
  [parse-eval-line (-> string? (hash/c string? real?))]
  [cuda-available? (-> boolean?)])
 dmatrix?
 booster?)

(module+ main
  (printf "xgboost version: ~a\n" (xgboost-version))
  (define dtrain
    (make-dmatrix '((1.0 2.0 0.5)
                    (2.0 1.0 1.5)
                    (3.0 0.5 0.0)
                    (0.5 3.0 2.0))
                  #:labels '(3.5 3.5 6.5 2.0)))
  (define b
    (train dtrain
           #:objective "reg:squarederror"
           #:max-depth 2
           #:eta 0.2
           #:verbosity 0
           #:rounds 5))
  (printf "first prediction: ~a\n" (car (predict b dtrain))))
