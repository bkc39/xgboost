#lang racket/base

;; Runner + tests for the literate example ../05-train-with-eval.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require racket/format
         racket/list
         xgboost
         "../05-train-with-eval.rkt")

(module+ main
  (define-values (booster history) (run-example))
  (define (col s) (~a s #:width 10 #:align 'right))
  (define (fmt v) (~r v #:precision '(= 4) #:min-width 10))
  (printf "~a  ~a  ~a\n" (~a "iter" #:width 4) (col "train-rmse") (col "eval-rmse"))
  (for ([m (in-list history)] [iter (in-naturals)])
    (printf "~a  ~a  ~a\n"
            (~a iter #:width 4)
            (fmt (hash-ref m "train-rmse"))
            (fmt (hash-ref m "eval-rmse"))))
  (define final (last history))
  (printf "\n~a boosted rounds\n" (booster-boosted-rounds booster))
  (printf "final train-rmse: ~a\n" (~r (hash-ref final "train-rmse") #:precision '(= 6)))
  (printf "final  eval-rmse: ~a\n" (~r (hash-ref final "eval-rmse") #:precision '(= 6))))

(module+ test
  (require rackunit)
  (define-values (booster history) (run-example))
  (check-equal? (booster-boosted-rounds booster) 30)
  (check-equal? (length history) 30)
  (define first-eval (hash-ref (first history) "eval-rmse"))
  (define final-eval (hash-ref (last history) "eval-rmse"))
  ;; Boosting should reduce the held-out error from where it started.
  (check-true (< final-eval first-eval)
              (format "eval-rmse did not improve: ~a -> ~a" first-eval final-eval)))
