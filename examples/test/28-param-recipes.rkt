#lang racket/base

;; Runner + tests for the literate example ../28-param-recipes.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../28-param-recipes.rkt")

(module+ main
  (define r (run-example))
  (for ([row (in-list (hash-ref r 'summary))])
    (printf "~a: boosted-rounds=~a  pred[0]=~a\n"
            (car row) (cadr row) (real->decimal-string (caddr row) 3)))
  (printf "monotone sweep: ~a violations (expected 0)\n" (hash-ref r 'violations)))

(module+ test
  (require rackunit)
  (define r (run-example))
  (for ([row (in-list (hash-ref r 'summary))])
    (check-true (cadddr row) (format "~a produced finite predictions" (car row))))
  (check-equal? (hash-ref r 'violations) 0
                "monotone constraint should hold across the feature-0 sweep"))
