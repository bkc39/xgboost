#lang racket/base

;; Runner + tests for the literate example ../19-booster-dumps-feature-scores.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../19-booster-dumps-feature-scores.rkt")

(module+ main
  (define result (run-example))
  (printf "booster feature names: ~a\n" (hash-ref result 'feature-info))
  (printf "dump count: ~a\n" (hash-ref result 'text-dump-count))
  (printf "score features: ~a\n" (hash-ref result 'score-features))
  (printf "score values: ~a\n" (hash-ref result 'score-values)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-equal? (hash-ref result 'feature-info) '("x0" "x1" "x2"))
  (check-true (> (hash-ref result 'text-dump-count) 0))
  (check-true (hash-ref result 'json-dump-has-object?))
  (check-true (hash-ref result 'named-dump-mentions-feature?))
  (check-true (pair? (hash-ref result 'score-features)))
  (check-true (pair? (hash-ref result 'score-shape)))
  (check-true (andmap positive? (hash-ref result 'score-values))))
