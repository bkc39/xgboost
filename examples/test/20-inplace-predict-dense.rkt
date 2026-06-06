#lang racket/base

;; Runner + tests for the literate example ../20-inplace-predict-dense.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require ffi/vector
         "../20-inplace-predict-dense.rkt")

(module+ main
  (define result (run-example))
  (printf "dense inplace predictions: ~a\n" (hash-ref result 'prediction-count))
  (printf "matches DMatrix prediction: ~a\n" (hash-ref result 'matches-dmatrix?)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (define inplace (hash-ref result 'inplace-preds))
  (define dmatrix (hash-ref result 'dmatrix-preds))
  (define diffs
    (for/list ([i (in-range (f32vector-length inplace))])
      (abs (- (f32vector-ref inplace i) (f32vector-ref dmatrix i)))))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true (hash-ref result 'matches-dmatrix?)
              (format "inplace vs dmatrix differ; max diff ~a" (apply max diffs))))
