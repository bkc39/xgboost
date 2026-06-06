#lang racket/base

;; Runner + tests for the literate example ../21-inplace-predict-csr.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../21-inplace-predict-csr.rkt")

(module+ main
  (define result (run-example))
  (printf "CSR inplace predictions: ~a\n" (hash-ref result 'prediction-count))
  (printf "matches DMatrix prediction: ~a\n" (hash-ref result 'matches-dmatrix?)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true (hash-ref result 'matches-dmatrix?)))
