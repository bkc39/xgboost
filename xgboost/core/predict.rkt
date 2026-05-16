#lang racket/base

;; High-level prediction.
;;
;; `predict` runs against a DMatrix; the `predict-from-*` variants take raw
;; Racket data (nested rows, CSR triples, columnar f32vectors) for
;; serving-style inference.  `#:as` selects the result shape — a plain list
;; (default) or the underlying f32vector.

(require ffi/vector
         (prefix-in foreign: "../foreign.rkt")
         "coerce.rkt")

(provide predict
         predict-from-dense
         predict-from-csr
         predict-from-columnar)

(define (preds->shape preds as)
  (case as
    [(f32vector) preds]
    [(list) (f32vector->list preds)]
    [else (raise-argument-error 'predict "'list or 'f32vector" as)]))

(define (predict b dmat
                 #:output [output 'value]
                 #:iteration-end [iteration-end 0]
                 #:as [as 'list])
  (define preds
    (foreign:booster-predict b dmat
                             #:output output
                             #:iteration-end iteration-end))
  (preds->shape preds as))

(define (predict-from-dense b data
                            #:nrow [nrow #f]
                            #:ncol [ncol #f]
                            #:missing [missing +nan.0]
                            #:output [output 'value]
                            #:iteration-end [iter-end 0]
                            #:as [as 'list])
  (define-values (vec rows cols) (coerce-matrix data nrow ncol))
  (preds->shape
   (foreign:booster-predict-from-dense b vec rows cols
                                       #:missing missing
                                       #:output output
                                       #:iteration-end iter-end)
   as))

(define (predict-from-csr b indptr indices data ncol
                          #:missing [missing +nan.0]
                          #:output [output 'value]
                          #:iteration-end [iter-end 0]
                          #:as [as 'list])
  (preds->shape
   (foreign:booster-predict-from-csr b indptr indices data ncol
                                     #:missing missing
                                     #:output output
                                     #:iteration-end iter-end)
   as))

(define (predict-from-columnar b columns
                               #:missing [missing +nan.0]
                               #:output [output 'value]
                               #:iteration-end [iter-end 0]
                               #:as [as 'list])
  (preds->shape
   (foreign:booster-predict-from-columnar b columns
                                          #:missing missing
                                          #:output output
                                          #:iteration-end iter-end)
   as))
