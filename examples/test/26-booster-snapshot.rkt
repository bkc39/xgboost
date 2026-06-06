#lang racket/base

;; Runner + tests for the literate example ../26-booster-snapshot.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../26-booster-snapshot.rkt")

(module+ main
  (define result (run-example))
  (printf "snapshot size: ~a bytes\n" (hash-ref result 'snapshot-bytes))
  (printf "matches immediately?     ~a\n" (hash-ref result 'matches-immediately?))
  (printf "matches after resume?    ~a\n" (hash-ref result 'matches-after-resume?)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-true (> (hash-ref result 'snapshot-bytes) 0))
  (check-true (hash-ref result 'matches-immediately?))
  (check-true (hash-ref result 'matches-after-resume?)))
