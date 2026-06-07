#lang racket/base

;; Runner + tests for the literate example ../09-poisson-bikes.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../09-poisson-bikes.rkt")

(define (col s) (~a s #:width 12 #:align 'right))
(define (fmt v) (~r v #:precision '(= 2) #:min-width 12))

(define (print-row name h)
  (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a name #:width 24)
          (fmt (hash-ref h 'rmse)) (fmt (hash-ref h 'mae))
          (fmt (hash-ref h 'min)) (col (hash-ref h 'neg))))

(module+ main
  (define r (run-example))
  (printf "final Poisson eval line:\n  ~a\n" (hash-ref r 'final-line))
  (printf "\ntest-set comparison (~a held-out days):\n" (hash-ref r 'n-test))
  (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a "model" #:width 24)
          (col "test RMSE") (col "test MAE") (col "min pred") (col "neg preds"))
  (print-row "count:poisson" (hash-ref r 'poisson))
  (print-row "reg:squarederror" (hash-ref r 'gaussian))
  (printf "\nfirst 12 test days (actual / poisson / gaussian):\n")
  (printf "  ~a  ~a  ~a  ~a\n" (~a "i" #:width 3)
          (col "actual") (col "poisson") (col "gaussian"))
  (for ([row (in-list (hash-ref r 'sample))] [i (in-naturals)])
    (printf "  ~a  ~a  ~a  ~a\n" (~a i #:width 3)
            (fmt (car row)) (fmt (cadr row)) (fmt (caddr row)))))

(module+ test
  (require rackunit)
  (define r (run-example))
  (check-equal? (hash-ref r 'n-test) 146)
  ;; Poisson predicts via exp(margin): structurally nonnegative.
  (check-equal? (hash-ref (hash-ref r 'poisson) 'neg) 0)
  (check-true (>= (hash-ref (hash-ref r 'poisson) 'min) 0.0)))
