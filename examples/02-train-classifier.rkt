#lang racket/base

;; End-to-end Racket-driven binary-classification training run.
;;
;; Two well-separated clusters in 4-d space; class 0 is small-valued, class 1
;; is large-valued.  Trains binary:logistic, prints predicted probabilities
;; alongside the truth labels, and reports accuracy at the 0.5 threshold.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/02-train-classifier.rkt

(require ffi/vector
         racket/format
         xgboost)

;; 10 rows x 4 features, alternating class 0 / class 1.
(define features
  (f32vector 0.1 0.2 0.1 0.0
             5.0 4.0 5.5 6.0
             0.3 0.5 0.1 0.2
             4.5 5.0 4.0 5.5
             0.0 0.1 0.2 0.0
             6.0 5.5 6.5 5.0
             0.4 0.3 0.2 0.5
             5.5 6.0 4.5 5.0
             0.2 0.1 0.3 0.1
             4.0 4.5 5.0 4.0))

(define labels (f32vector 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))

(define dtrain
  (make-dmatrix features #:nrow 10 #:ncol 4 #:labels labels))

(printf "training data:\n")
(dmatrix-show dtrain)
(newline)

(define rounds 30)
(define booster
  (train dtrain
         #:objective "binary:logistic"
         #:max-depth 3
         #:eta 0.3
         #:verbosity 0
         #:rounds rounds))

(define probs (predict booster dtrain #:as 'f32vector))

(define (fmt v) (~r v #:precision '(= 4) #:min-width 8))
(define (col s) (~a s #:width 8 #:align 'right))

(printf "predictions vs. labels (~a rounds):\n" rounds)
(printf "  ~a  ~a  ~a  ~a\n"
        (~a "i" #:width 3) (col "truth") (col "p(1)") (col "pred"))

(define n (f32vector-length labels))
(define correct
  (for/sum ([i (in-range n)])
    (define truth (inexact->exact (round (f32vector-ref labels i))))
    (define p (f32vector-ref probs i))
    (define pred (if (> p 0.5) 1 0))
    (printf "  ~a  ~a  ~a  ~a\n"
            (~a i #:width 3)
            (col truth)
            (fmt p)
            (col pred))
    (if (= pred truth) 1 0)))

(printf "\naccuracy: ~a/~a (~a%)\n"
        correct n
        (~r (* 100 (/ correct n)) #:precision '(= 1)))
