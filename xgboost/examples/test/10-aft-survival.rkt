#lang racket/base

;; Runner + tests for the literate example ../10-aft-survival.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         racket/list
         "../10-aft-survival.rkt")

(define (log-iter? i) (or (< i 5) (zero? (modulo (add1 i) 25))))

(module+ main
  (define r (run-example))
  (printf "train: ~a rows (~a right-censored), test: ~a rows\n"
          (hash-ref r 'n-train) (hash-ref r 'n-censored-train) (hash-ref r 'n-test))
  (printf "\ntraining (watching aft-nloglik):\n")
  (printf "  ~a  ~a  ~a\n" (~a "iter" #:width 4)
          (~a "train-aft-nloglik" #:width 18 #:align 'right)
          (~a "test-aft-nloglik" #:width 18 #:align 'right))
  (for ([m (in-list (hash-ref r 'history))] [iter (in-naturals)] #:when (log-iter? iter))
    (printf "  ~a  ~a  ~a\n" (~a iter #:width 4)
            (~r (hash-ref m "train-aft-nloglik") #:precision '(= 4) #:min-width 18)
            (~r (hash-ref m "test-aft-nloglik") #:precision '(= 4) #:min-width 18)))
  (define (col s) (~a s #:width 12 #:align 'right))
  (define (fmt v) (~r v #:precision '(= 1) #:min-width 12))
  (printf "\npredictions on the held-out test set:\n")
  (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a "i" #:width 3)
          (col "lower") (col "upper") (col "censored?") (col "predicted"))
  (for ([row (in-list (hash-ref r 'rows))] [i (in-naturals)])
    (printf "  ~a  ~a  ~a  ~a  ~a\n" (~a i #:width 3)
            (fmt (car row))
            (if (caddr row) (~a "+inf" #:width 12 #:align 'right) (fmt (cadr row)))
            (~a (if (caddr row) "yes" "no") #:width 12 #:align 'right)
            (fmt (cadddr row)))))

(module+ test
  (require rackunit)
  (define r (run-example))
  (check-equal? (length (hash-ref r 'history)) 100)
  ;; AFT loss should fall over training.
  (define hist (hash-ref r 'history))
  (check-true (< (hash-ref (last hist) "test-aft-nloglik")
                 (hash-ref (first hist) "test-aft-nloglik")))
  ;; AFT predict = exp(margin): every prediction is finite and positive.
  (check-equal? (hash-ref r 'n-positive) (hash-ref r 'n-test))
  (check-equal? (hash-ref r 'n-finite) (hash-ref r 'n-test)))
