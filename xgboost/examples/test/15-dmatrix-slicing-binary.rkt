#lang racket/base

;; Runner + tests for the literate example ../15-dmatrix-slicing-binary.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require "../15-dmatrix-slicing-binary.rkt")

(define (same-matrix? expected got)
  (and (= (length expected) (length got))
       (for/and ([er (in-list expected)] [gr (in-list got)])
         (and (= (length er) (length gr))
              (for/and ([e (in-list er)] [g (in-list gr)]) (< (abs (- g e)) 1e-6))))))

(module+ main
  (define result (run-example))
  (define loaded (hash-ref result 'loaded-summary))
  (printf "sliced values: ~a\n" (hash-ref result 'sliced-values))
  (printf "loaded binary shape: ~ax~a\n"
          (hash-ref loaded 'rows) (hash-ref loaded 'cols)))

(module+ test
  (require rackunit)
  (define expected '((5.0 6.0) (1.0 2.0)))
  (define result (run-example))
  (define loaded (hash-ref result 'loaded-summary))
  (check-true (same-matrix? expected (hash-ref result 'sliced-values)))
  (check-equal? (hash-ref loaded 'rows) 2)
  (check-equal? (hash-ref loaded 'cols) 2)
  (check-true (same-matrix? expected (hash-ref loaded 'values))))
