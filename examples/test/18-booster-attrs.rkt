#lang racket/base

;; Runner + tests for the literate example ../18-booster-attrs.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../18-booster-attrs.rkt")

(module+ main
  (define result (run-example))
  (define before (hash-ref result 'before-delete))
  (printf "owner: ~a\n" (hash-ref before 'owner))
  (printf "attrs before delete: ~a\n" (hash-ref before 'names))
  (printf "purpose after delete: ~a\n" (hash-ref result 'purpose-after-delete)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (define before (hash-ref result 'before-delete))
  (check-equal? (hash-ref before 'owner) "racket")
  (check-equal? (hash-ref before 'purpose) "example")
  (check-equal? (hash-ref before 'names) '("owner" "purpose"))
  (check-false (hash-ref result 'purpose-after-delete))
  (check-equal? (hash-ref result 'names-after-delete) '("owner")))
