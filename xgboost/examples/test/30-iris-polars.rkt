#lang racket/base

;; Runner + tests for the literate example ../30-iris-polars.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../30-iris-polars.rkt")

(module+ main
  (define-values (n-train n-test acc parquet-ok?) (run-example))
  (printf "iris-polars: ~a train / ~a test, test accuracy ~a, parquet round-trip ~a\n"
          n-train n-test (~r acc #:precision '(= 3))
          (if parquet-ok? "ok" "MISMATCH")))

(module+ test
  (require rackunit)
  (define-values (n-train n-test acc parquet-ok?) (run-example))
  (check-equal? n-train 120)
  (check-equal? n-test 30)
  ;; Iris is easily separable; this floor is comfortably cleared.
  (check-true (> acc 0.8) (format "expected accuracy > 0.8, got ~a" acc))
  (check-true parquet-ok? "Parquet round-trip changed the predictions"))
