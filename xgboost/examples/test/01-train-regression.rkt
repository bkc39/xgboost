#lang racket/base

;; Runner + tests for the literate example ../01-train-regression.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require ffi/vector
         racket/format
         xgboost
         "../01-train-regression.rkt")

;; Mean squared error between f32vector predictions and a list of labels.
(define (mse preds labels)
  (define n (f32vector-length preds))
  (define ls (list->vector labels))
  (/ (for/sum ([i (in-range n)])
       (define d (- (f32vector-ref preds i) (vector-ref ls i)))
       (* d d))
     n))

(module+ main
  (define-values (booster dtrain preds) (run-example))
  (define labels (dmatrix-label dtrain))
  (define n (f32vector-length preds))
  (define (fmt v) (~r v #:precision '(= 4) #:min-width 8))
  (printf "predictions vs. labels:\n")
  (printf "  ~a  ~a  ~a\n" (~a "i" #:width 3) (~a "label" #:width 8 #:align 'right)
          (~a "pred" #:width 8 #:align 'right))
  (for ([i (in-range n)])
    (printf "  ~a  ~a  ~a\n"
            (~a i #:width 3)
            (fmt (list-ref labels i))
            (fmt (f32vector-ref preds i))))
  (printf "\nmodel: ~a boosted rounds\n" (booster-boosted-rounds booster))
  (printf "training MSE: ~a\n" (~r (mse preds labels) #:precision '(= 6))))

(module+ test
  (require rackunit)
  (define-values (booster dtrain preds) (run-example))
  (check-pred booster? booster)
  (check-equal? (booster-boosted-rounds booster) 50)
  (check-equal? (f32vector-length preds) 8)
  ;; 50 rounds on this easy synthetic target should fit tightly.
  (check-true (< (mse preds (dmatrix-label dtrain)) 1.0)))
