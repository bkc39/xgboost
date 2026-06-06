#lang racket/base

;; Runner + tests for the literate example ../17-booster-lifecycle-config.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../17-booster-lifecycle-config.rkt")

(module+ main
  (define result (run-example))
  (printf "boosted rounds (after reset): ~a\n" (hash-ref result 'rounds))
  (printf "num features: ~a\n" (hash-ref result 'features))
  (printf "sliced rounds: ~a\n" (hash-ref result 'sliced-rounds))
  (printf "config JSON: ~a\n" (hash-ref result 'config-json?)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-equal? (hash-ref result 'rounds) 8)
  (check-equal? (hash-ref result 'features) 3)
  (check-equal? (hash-ref result 'sliced-rounds) 3)
  (check-true (hash-ref result 'config-json?)))
