#lang racket/base

;; Runner + tests for the literate example ../02-train-classifier.rkt.
;; The example is a #lang scribble/lp2 program; main/test live here and require
;; its `run-example` provide (lp2 submodules can't see chunk-level bindings).

(require ffi/vector
         racket/format
         xgboost
         "../02-train-classifier.rkt")

;; Hard predictions and correct-count at the 0.5 threshold.
(define (score probs labels)
  (define n (f32vector-length labels))
  (for/sum ([i (in-range n)])
    (define truth (inexact->exact (round (f32vector-ref labels i))))
    (define pred (if (> (f32vector-ref probs i) 0.5) 1 0))
    (if (= pred truth) 1 0)))

(module+ main
  (define-values (booster dtrain probs) (run-example))
  (define labels (dmatrix-label dtrain))
  (define n (length labels))
  (define (fmt v) (~r v #:precision '(= 4) #:min-width 8))
  (define (col s) (~a s #:width 8 #:align 'right))
  (printf "predictions vs. labels:\n")
  (printf "  ~a  ~a  ~a  ~a\n"
          (~a "i" #:width 3) (col "truth") (col "p(1)") (col "pred"))
  (for ([i (in-range n)])
    (define p (f32vector-ref probs i))
    (printf "  ~a  ~a  ~a  ~a\n"
            (~a i #:width 3)
            (col (inexact->exact (round (list-ref labels i))))
            (fmt p)
            (col (if (> p 0.5) 1 0))))
  (define correct (score probs (list->f32vector labels)))
  (printf "\nmodel: ~a boosted rounds\n" (booster-boosted-rounds booster))
  (printf "accuracy: ~a/~a (~a%)\n"
          correct n (~r (* 100 (/ correct n)) #:precision '(= 1))))

(module+ test
  (require rackunit)
  (define-values (booster dtrain probs) (run-example))
  (check-equal? (booster-boosted-rounds booster) 30)
  (check-equal? (f32vector-length probs) 10)
  ;; The two clusters are linearly separable; the model should be perfect.
  (check-equal? (score probs (list->f32vector (dmatrix-label dtrain))) 10))
