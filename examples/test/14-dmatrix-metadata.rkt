#lang racket/base

;; Runner + tests for the literate example ../14-dmatrix-metadata.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../14-dmatrix-metadata.rkt")

(module+ main
  (define result (run-example))
  (printf "feature names: ~a\n" (hash-ref result 'feature-names))
  (printf "feature types: ~a\n" (hash-ref result 'feature-types))
  (printf "group ptr: ~a\n" (hash-ref result 'group-ptr))
  (printf "labels: ~a\n" (hash-ref result 'labels)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-equal? (hash-ref result 'feature-names) '("height" "weight"))
  (check-equal? (hash-ref result 'feature-types) '("q" "q"))
  (check-equal? (hash-ref result 'group-ptr) '(0 2))
  (check-equal? (hash-ref result 'labels) '(0.25 0.75)))
