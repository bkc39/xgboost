#lang racket/base

;; Runner + tests for the literate example ../16-quantile-cuts.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require ffi/vector
         racket/list
         "../16-quantile-cuts.rkt")

(module+ main
  (define result (run-example))
  (printf "quantile indptr length: ~a\n" (hash-ref result 'indptr-length))
  (printf "quantile data length: ~a\n" (hash-ref result 'data-length))
  (printf "prediction count: ~a\n" (hash-ref result 'prediction-count)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (define indptr (hash-ref result 'indptr))
  (define data (hash-ref result 'data))
  (check-true (positive? (length indptr)))
  (check-equal? (first indptr) 0)
  (check-true (positive? (last indptr)))
  (check-true (f32vector? data))
  (check-equal? (f32vector-length data) (last indptr))
  (check-equal? (hash-ref result 'prediction-count) 4))
