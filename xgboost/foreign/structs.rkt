#lang racket/base

;; Struct wrappers.
;;
;; DMatrix and Booster are wrapper structs over the raw cpointers from
;; foreign/raw.  Both use `prop:cpointer` so they are accepted directly by
;; typed FFI calls (`_DMatrix`, `_Booster`) — no manual handle extraction is
;; needed when threading them through the raw layer.
;;
;; The cpointer is the first field, so prop:cpointer 0 makes the struct
;; itself stand in for the cpointer.  The struct retains additional Racket
;; data:
;;   dmatrix:  cached row/col counts (avoids a C round-trip per query).
;;   booster:  the list of DMatrix wrappers passed at creation time.  XGBoost
;;             stores those handles as weak references for prediction-cache
;;             invalidation, so they must outlive the booster — keeping them
;;             in a struct field pins their lifetime to ours, no global
;;             retention table required.
;;
;; The auto-generated struct predicate matches any wrapper, alive or not.
;; The exported `dmatrix?` / `booster?` predicates additionally require the
;; underlying cpointer to still carry its DMatrix / Booster tag, so a wrapper
;; whose handle has been explicitly freed (via the `unsafe` submodule) fails
;; contract checks with a clearer message than the raw FFI tag-mismatch error.
;;
;; The lifetime model: the raw layer's `#:wrap (allocator ...)` auto-registers
;; a finalizer that calls the matching `*-free/raw` deallocator.  An explicit
;; `*-free!` call runs the deallocator — which cancels the finalizer so the
;; free does not execute twice — and flips the cpointer tag so a second
;; explicit free raises `exn:fail:contract` at the contract boundary instead
;; of double-freeing at the C level.

(require ffi/unsafe
         "raw/dmatrix.rkt"
         "raw/booster.rkt"
         "error.rkt")

(provide (struct-out dmatrix-impl)
         dmatrix?
         dmatrix-handle
         dmatrix-rows
         dmatrix-cols
         wrap-dmatrix
         dmatrix-free!
         (struct-out booster-impl)
         booster?
         booster-handle
         booster-cache
         booster-free!)

(struct dmatrix-impl (handle rows cols)
  #:reflection-name 'dmatrix
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (dm port _mode)
    (fprintf port "#<dmatrix:~ax~a>"
             (dmatrix-impl-rows dm)
             (dmatrix-impl-cols dm))))

(define (dmatrix? v)
  (and (dmatrix-impl? v) (DMatrix? v)))

(define dmatrix-handle dmatrix-impl-handle)
(define dmatrix-rows dmatrix-impl-rows)
(define dmatrix-cols dmatrix-impl-cols)

(define (wrap-dmatrix h)
  (define-values (rc-r nrow) (xgb-dmatrix-num-row/raw h))
  (check-ok rc-r 'wrap-dmatrix)
  (define-values (rc-c ncol) (xgb-dmatrix-num-col/raw h))
  (check-ok rc-c 'wrap-dmatrix)
  (dmatrix-impl h nrow ncol))

(define (dmatrix-free! dm)
  (define h (dmatrix-handle dm))
  (when (cpointer-has-tag? h 'DMatrix)
    (xgb-dmatrix-free/raw h)
    (set-cpointer-tag! h 'DMatrix-freed)))

(struct booster-impl (handle cache)
  #:reflection-name 'booster
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (_b port _mode)
    (fprintf port "#<booster>")))

(define (booster? v)
  (and (booster-impl? v) (Booster? v)))

(define booster-handle booster-impl-handle)
(define booster-cache booster-impl-cache)

(define (booster-free! b)
  (define h (booster-handle b))
  (when (cpointer-has-tag? h 'Booster)
    (xgb-booster-free/raw h)
    (set-cpointer-tag! h 'Booster-freed)))
