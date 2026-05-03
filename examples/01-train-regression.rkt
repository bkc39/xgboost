#lang racket/base

;; End-to-end Racket-driven regression training run.
;;
;; Build a small synthetic dataset, train a gradient-boosted regressor, and
;; print predictions vs. labels alongside training MSE.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/01-train-regression.rkt

(require ffi/vector
         racket/format
         xgboost/ffi)

;; 8 rows x 3 features.  Labels were chosen to be roughly
;; `2*x0 + x1 - x2` plus a touch of noise — a tree booster fits this easily.
(define features
  (f32vector 1.0 2.0 0.5
             2.0 1.0 1.5
             3.0 0.5 0.0
             0.5 3.0 2.0
             4.0 2.0 1.0
             1.5 1.5 0.5
             2.5 3.5 1.5
             0.0 1.0 0.0))

(define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define dtrain (dmatrix-create-from-mat features 8 3))
(dmatrix-set-float-info! dtrain "label" labels)

(printf "training data:\n")
(dmatrix-show dtrain)
(newline)

(define booster (booster-create (list dtrain)))
(booster-set-param! booster "objective"  "reg:squarederror")
(booster-set-param! booster "max_depth"  "3")
(booster-set-param! booster "eta"        "0.1")
(booster-set-param! booster "verbosity"  "0")

(define rounds 50)
(for ([iter (in-range rounds)])
  (booster-update-one-iter! booster iter dtrain))

(define preds (booster-predict booster dtrain))

(define (fmt v) (~r v #:precision '(= 4) #:min-width 8))
(define (col s) (~a s #:width 8 #:align 'right))

(printf "predictions vs. labels (~a rounds):\n" rounds)
(printf "  ~a  ~a  ~a\n" (~a "i" #:width 3) (col "label") (col "pred"))
(define n (f32vector-length labels))
(for ([i (in-range n)])
  (printf "  ~a  ~a  ~a\n"
          (~a i #:width 3)
          (fmt (f32vector-ref labels i))
          (fmt (f32vector-ref preds i))))

(define mse
  (/ (for/sum ([i (in-range n)])
       (define d (- (f32vector-ref preds i) (f32vector-ref labels i)))
       (* d d))
     n))

(printf "\ntraining MSE: ~a\n" (~r mse #:precision '(= 6)))

(booster-free! booster)
(dmatrix-free! dtrain)
