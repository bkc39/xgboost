#lang racket/base

;; Runner + tests for the literate example ../24-cuda-regression.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its provides (lp2 submodules can't see chunk-level bindings). CUDA work is
;; gated on `cuda-available?`, so this skips gracefully on CPU-only builds.

(require racket/format
         "../24-cuda-regression.rkt")

(module+ main
  (cond
    [(cuda-available?)
     (define result (run-example))
     (printf "CUDA regression predictions: ~a\n" (hash-ref result 'prediction-count))
     (printf "training MSE: ~a\n" (~r (hash-ref result 'mse) #:precision '(= 6)))
     (printf "MSE < 1.0: ~a\n" (hash-ref result 'improved?))]
    [else (printf "CUDA not available in this build; skipping.\n")]))

(module+ test
  (require rackunit)
  (when (cuda-available?)
    (define result (run-example))
    (check-equal? (hash-ref result 'prediction-count) 8)
    (check-true (hash-ref result 'improved?))
    (check-true (< (hash-ref result 'mse) 1.0))))
