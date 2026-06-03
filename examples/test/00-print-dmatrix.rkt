#lang racket/base

;; Runner + tests for the literate example ../00-print-dmatrix.rkt.
;; The example itself is a #lang scribble/lp2 program (woven into the manual);
;; lp2 submodules can't see chunk-level bindings, so main/test live here and
;; require the example's provides.

(require racket/pretty
         xgboost
         "../00-print-dmatrix.rkt")

(module+ main
  (define dm (run-example))
  (dmatrix-show dm)
  (newline)
  (pretty-print (dmatrix->list dm)))

(module+ test
  (require rackunit)
  (define dm (run-example))
  (check-equal? (dmatrix-rows dm) 4)
  (check-equal? (dmatrix-cols dm) 3)
  (check-equal? (dmatrix-label dm) '(3.5 3.5 6.5 2.0)))
