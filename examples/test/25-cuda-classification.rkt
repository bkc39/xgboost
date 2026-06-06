#lang racket/base

;; Runner + tests for the literate example ../25-cuda-classification.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its provides (lp2 submodules can't see chunk-level bindings). CUDA work is
;; gated on `cuda-available?`, so this skips gracefully on CPU-only builds.

(require racket/format
         "../25-cuda-classification.rkt")

(module+ main
  (cond
    [(cuda-available?)
     (define result (run-example))
     (printf "CUDA classification predictions: ~a\n" (hash-ref result 'prediction-count))
     (printf "accuracy: ~a/~a (~a%)\n"
             (hash-ref result 'correct) (hash-ref result 'prediction-count)
             (~r (* 100 (hash-ref result 'accuracy)) #:precision '(= 1)))
     (printf "perfect separation: ~a\n" (hash-ref result 'perfect?))]
    [else (printf "CUDA not available in this build; skipping.\n")]))

(module+ test
  (require rackunit)
  (when (cuda-available?)
    (define result (run-example))
    (check-equal? (hash-ref result 'prediction-count) 10)
    (check-true (hash-ref result 'all-valid-probs?))
    (check-true (hash-ref result 'perfect?))))
