#lang racket/base

;; Booster lifecycle and training-step primitives: creation, reset, slicing,
;; shape queries, parameter setting, and the per-iteration update calls.

(require ffi/vector
         "../raw/global.rkt"
         "../raw/booster.rkt"
         "../structs.rkt"
         "../error.rkt"
         "../array-interface.rkt")

(provide booster-create
         booster-reset!
         booster-slice
         booster-boosted-rounds
         booster-num-feature
         booster-set-param!
         booster-update-one-iter!
         booster-train-one-iter!)

;; XGBoost's XGBoosterCreate stores the cache DMatrix handles for prediction
;; result caching and expects them to outlive the booster.  At the C level it
;; only keeps weak references — so if Racket GCs a cache DMatrix while the
;; booster still exists, subsequent operations would touch freed memory.  The
;; booster struct's cache field pins those DMatrices alive for the booster's
;; lifetime; no out-of-band retention table is needed.
(define (booster-create [cache '()])
  (define h (xgb-booster-create/raw (map dmatrix-handle cache)))
  (unless h
    (error 'booster-create
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  (booster-impl h cache))

(define (booster-reset! b)
  (check-ok (xgb-booster-reset/raw b) 'booster-reset!))

(define (booster-slice b begin-layer end-layer [step 1])
  (define sliced (xgb-booster-slice/raw b begin-layer end-layer step))
  (unless sliced
    (error 'booster-slice
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  ;; A sliced booster is C-side independent and needs no Racket-side cache.
  (booster-impl sliced '()))

(define (booster-boosted-rounds h)
  (define-values (rc out) (xgb-booster-boosted-rounds/raw h))
  (check-ok rc 'booster-boosted-rounds)
  out)

(define (booster-num-feature h)
  (define-values (rc out) (xgb-booster-num-feature/raw h))
  (check-ok rc 'booster-num-feature)
  out)

(define (booster-set-param! h key value)
  (check-ok (xgb-booster-set-param/raw h key value) 'booster-set-param!))

(define (booster-update-one-iter! h iter dtrain)
  (check-ok (xgb-booster-update-one-iter/raw h iter dtrain)
            'booster-update-one-iter!))

(define (booster-train-one-iter! h iter dtrain grad hess)
  (unless (= (f32vector-length grad) (f32vector-length hess))
    (error 'booster-train-one-iter!
           "gradient length ~a does not match Hessian length ~a"
           (f32vector-length grad)
           (f32vector-length hess)))
  (check-ok (xgb-booster-train-one-iter/raw
             h
             dtrain
             iter
             (f32-array-interface grad (list (f32vector-length grad)))
             (f32-array-interface hess (list (f32vector-length hess))))
            'booster-train-one-iter!))
