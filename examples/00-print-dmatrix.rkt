#lang racket/base

;; Smallest possible end-to-end exercise of the DMatrix primitives.
;; Create a 4x3 matrix of features, attach labels + weights, print.
;; The Racket GC frees the underlying handle when `dm` becomes unreachable.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/00-print-dmatrix.rkt

(require ffi/vector
         racket/pretty
         xgboost)

(define features
  (f32vector 1.0 2.0 0.5
             2.0 1.0 1.5
             3.0 0.5 0.0
             0.5 3.0 2.0))

(define labels (f32vector 3.5 3.5 6.5 2.0))
(define weights (f32vector 1.0 1.0 2.0 0.5))

(define dm
  (make-dmatrix features
                #:nrow 4
                #:ncol 3
                #:labels labels
                #:weights weights))

(dmatrix-show dm)
(newline)
(pretty-print (dmatrix->list dm))
