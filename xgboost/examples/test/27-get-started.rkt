#lang racket/base

;; Runner + tests for the literate example ../27-get-started.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../27-get-started.rkt")

(module+ main
  (define-values (n-train preds acc) (run-example))
  (printf "get-started: ~a train / ~a test, test accuracy ~a\n"
          n-train (length preds) (~r acc #:precision '(= 3))))

(module+ test
  (require rackunit)
  (define-values (n-train preds acc) (run-example))
  (check-equal? n-train 120)
  (check-equal? (length preds) 30)
  ;; Iris is easily separable; even two shallow rounds clear this floor.
  (check-true (> acc 0.8) (format "expected accuracy > 0.8, got ~a" acc)))
