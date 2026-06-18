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
         ;; DataFrame predicates for the polars bridge contracts.
         (only-in polars dataframe? series?)
         "core/global.rkt"
         ;; make-dmatrix/train/predict are wrapped below to accept a polars
         ;; DataFrame anywhere a DMatrix is expected; rename-in passes every
         ;; other identifier from these modules through unchanged.
         (rename-in "core/dmatrix.rkt" [make-dmatrix core:make-dmatrix])
         "core/dataframe.rkt"
         "core/booster.rkt"
         (rename-in "core/predict.rkt" [predict core:predict])
         (rename-in "core/train.rkt" [train core:train])
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
  ;; Polars DataFrame bridge (see core/dataframe.rkt).
  [dataframe->dmatrix
   (->* (dataframe?)
        (#:labels any/c
         #:weights any/c
         #:feature-columns (or/c #f (listof string?))
         #:missing real?)
        dmatrix?)]
  [series->f32vector (-> series? f32vector?)]
  [dataframe-column->f32vector
   (-> dataframe? (or/c string? exact-nonnegative-integer?) f32vector?)]
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
   (->* ((or/c dmatrix? dataframe?))
        (#:params any/c
         #:rounds exact-nonnegative-integer?
         #:evals (listof (cons/c string? dmatrix?))
         #:objective (or/c #f any/c)
         #:objective-fn (or/c #f (-> f32vector? dmatrix? any))
         #:eta (or/c #f any/c)
         #:max-depth (or/c #f any/c)
         #:num-class (or/c #f any/c)
         #:eval-metric (or/c #f any/c)
         #:verbosity (or/c #f any/c)
         ;; DataFrame path only: column name (or sequence) for the labels.
         #:labels (or/c #f any/c))
        booster?)]
  [booster-set-param! (-> booster? any/c any/c void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-train-one-iter!
   (-> booster? exact-integer? dmatrix? any/c any/c void?)]
  [booster->bytes (-> booster? bytes?)]
  [bytes->booster (-> bytes? booster?)]
  [predict
   (->* (booster? (or/c dmatrix? dataframe?))
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

;; ---------------------------------------------------------------------------
;; DataFrame-aware facade wrappers
;;
;; The public make-dmatrix/train/predict accept a polars DataFrame anywhere a
;; DMatrix is expected, delegating to the core/dataframe.rkt bridge.  Dispatch
;; lives here at the facade (not in core/dmatrix.rkt) to keep the core require
;; graph one-directional: dataframe.rkt already requires dmatrix.rkt, so a
;; reverse dependency would form a cycle.

(define (make-dmatrix data
                      #:nrow [nrow #f]
                      #:ncol [ncol #f]
                      #:missing [missing +nan.0]
                      #:labels [labels #f]
                      #:weights [weights #f])
  (if (dataframe? data)
      (dataframe->dmatrix data
                          #:labels labels
                          #:weights weights
                          #:missing missing)
      (core:make-dmatrix data
                         #:nrow nrow
                         #:ncol ncol
                         #:missing missing
                         #:labels labels
                         #:weights weights)))

;; #:labels names the label column (or supplies a label sequence) when `data`
;; is a DataFrame; it is ignored on the DMatrix path, where labels already
;; ride on the matrix.
(define (train data
               #:params [params '()]
               #:rounds [rounds 10]
               #:evals [evals '()]
               #:objective [objective #f]
               #:objective-fn [objective-fn #f]
               #:eta [eta #f]
               #:max-depth [max-depth #f]
               #:num-class [num-class #f]
               #:eval-metric [eval-metric #f]
               #:verbosity [verbosity #f]
               #:labels [labels #f])
  (define dtrain
    (if (dataframe? data)
        (dataframe->dmatrix data #:labels labels)
        data))
  (core:train dtrain
              #:params params
              #:rounds rounds
              #:evals evals
              #:objective objective
              #:objective-fn objective-fn
              #:eta eta
              #:max-depth max-depth
              #:num-class num-class
              #:eval-metric eval-metric
              #:verbosity verbosity))

(define (predict b data
                 #:output [output 'value]
                 #:iteration-end [iteration-end 0]
                 #:as [as 'list])
  (define dpred
    (if (dataframe? data)
        (dataframe->dmatrix data)
        data))
  (core:predict b dpred
                #:output output
                #:iteration-end iteration-end
                #:as as))

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

(module+ test
  ;; The DataFrame-aware facade must be a transparent alias for the explicit
  ;; DMatrix path: training/predicting on a DataFrame yields bit-identical
  ;; results to building the DMatrix by hand.
  (require rackunit
           (only-in polars dataframe series))

  (define rows '((1.0 2.0 0.5)
                 (2.0 1.0 1.5)
                 (3.0 0.5 0.0)
                 (0.5 3.0 2.0)))
  (define ys '(3.5 3.5 6.5 2.0))

  ;; Feature columns a/b/c carry the same values as `rows`; the training frame
  ;; also carries the label column "y".  `feat-cols` is the matching
  ;; label-free frame used for prediction (you have no labels at predict time).
  (define a (series '(1.0 2.0 3.0 0.5) #:name "a"))
  (define b-col (series '(2.0 1.0 0.5 3.0) #:name "b"))
  (define c (series '(0.5 1.5 0.0 2.0) #:name "c"))
  (define df (dataframe (list a b-col c (series ys #:name "y"))))
  (define feat-df (dataframe (list a b-col c)))

  (define (fit dtrain)
    (train dtrain
           #:objective "reg:squarederror"
           #:max-depth 2
           #:eta 0.2
           #:verbosity 0
           #:rounds 5))

  (test-case "make-dmatrix dispatches on DataFrame"
    (define dm (make-dmatrix df #:labels "y"))
    (check-equal? (dmatrix-rows dm) 4)
    (check-equal? (dmatrix-cols dm) 3 "label column excluded from features")
    (check-equal? (dmatrix-label dm) ys))

  (test-case "train/predict on a DataFrame match the explicit DMatrix path"
    (define dm (make-dmatrix rows #:labels ys))
    (define b-dm (fit dm))
    (define b-df (fit (make-dmatrix df #:labels "y")))
    ;; The two boosters are trained on identical data, so predictions agree
    ;; whether the input is a DMatrix or a (label-free) DataFrame.
    (define p-dm (predict b-dm dm))
    (check-equal? (predict b-df feat-df) p-dm)
    (check-equal? (predict b-dm feat-df) p-dm
                  "predict on a DataFrame matches predict on the DMatrix"))

  (test-case "train accepts a DataFrame directly via #:labels"
    (define b-direct (train df
                            #:labels "y"
                            #:objective "reg:squarederror"
                            #:max-depth 2
                            #:eta 0.2
                            #:verbosity 0
                            #:rounds 5))
    (define b-explicit (fit (make-dmatrix df #:labels "y")))
    (check-equal? (predict b-direct feat-df) (predict b-explicit feat-df))))
