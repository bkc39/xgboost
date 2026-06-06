#lang racket/base

;; Runner + tests for the literate example ../07-robust-regression.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         "../07-robust-regression.rkt")

(module+ main
  (define r (run-example))
  (define (col s) (~a s #:width 12 #:align 'right))
  (define (fmt v) (~r v #:precision '(= 2) #:min-width 12))
  (printf "huber_slope sweep (MSE/MAE on clean test set):\n")
  (printf "  ~a  ~a  ~a\n" (col "huber_slope") (col "test MSE") (col "test MAE"))
  (for ([row (in-list (hash-ref r 'hub-table))])
    (printf "  ~a  ~a  ~a\n" (col (car row)) (fmt (cadr row)) (fmt (caddr row))))
  (printf "\nperformance on the clean test set (training has 10% outliers):\n")
  (printf "  ~a  ~a  ~a\n" (~a "loss" #:width 32) (col "test MSE") (col "test MAE"))
  (printf "  ~a  ~a  ~a\n" (~a "reg:squarederror" #:width 32)
          (fmt (hash-ref r 'sq-mse)) (fmt (hash-ref r 'sq-mae)))
  (printf "  ~a  ~a  ~a\n"
          (~a (format "reg:pseudohubererror δ=~a" (hash-ref r 'hub-slope)) #:width 32)
          (fmt (hash-ref r 'hub-mse)) (fmt (hash-ref r 'hub-mae)))
  (printf "  ~a  ~a  ~a\n" (~a "reg:absoluteerror" #:width 32)
          (fmt (hash-ref r 'l1-mse)) (fmt (hash-ref r 'l1-mae))))

(module+ test
  (require rackunit)
  (define r (run-example))
  (check-equal? (hash-ref r 'n-test) 89)
  ;; The whole point: robust losses generalize better than squared error when
  ;; the training labels are outlier-corrupted.
  (check-true (< (hash-ref r 'l1-mse) (hash-ref r 'sq-mse))
              "L1 should beat squared error on the clean test set")
  (check-true (<= (hash-ref r 'hub-mse) (hash-ref r 'sq-mse))
              "best Huber should beat squared error on the clean test set"))
