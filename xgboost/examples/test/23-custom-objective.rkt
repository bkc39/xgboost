#lang racket/base

;; Runner + tests for the literate example ../23-custom-objective.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../23-custom-objective.rkt")

(module+ main
  (define result (run-example))
  (printf "custom objective predictions: ~a\n" (hash-ref result 'prediction-count))
  (printf "initial MSE: ~a\n" (hash-ref result 'initial-mse))
  (printf "final MSE: ~a\n" (hash-ref result 'final-mse)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true (hash-ref result 'improved?))
  (check-true (< (hash-ref result 'final-mse) 3.0)))
