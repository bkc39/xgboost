#lang racket/base

;; Train a regressor while watching its loss on a held-out eval set.
;; Each round emits a metric line like:
;;   [iter] train-rmse:... eval-rmse:...
;; that we parse with `parse-eval-line` and print as a small table.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/05-train-with-eval.rkt

(require ffi/vector
         racket/format
         xgboost/ffi)

;; Two non-overlapping splits of an y ≈ 2*x0 + x1 - x2 dataset.

(define train-features
  (f32vector 1.0 2.0 0.5
             2.0 1.0 1.5
             3.0 0.5 0.0
             0.5 3.0 2.0
             4.0 2.0 1.0
             1.5 1.5 0.5
             2.5 3.5 1.5
             0.0 1.0 0.0))
(define train-labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define eval-features
  (f32vector 2.0 0.5 0.5
             1.0 1.0 1.0
             3.5 1.0 0.5
             0.5 0.5 0.5))
(define eval-labels (f32vector 4.0 2.0 7.5 1.0))

(define dtrain (dmatrix-create-from-mat train-features 8 3))
(dmatrix-set-float-info! dtrain "label" train-labels)

(define deval (dmatrix-create-from-mat eval-features 4 3))
(dmatrix-set-float-info! deval "label" eval-labels)

(define booster (booster-create (list dtrain)))
(booster-set-param! booster "objective" "reg:squarederror")
(booster-set-param! booster "max_depth" "3")
(booster-set-param! booster "eta"       "0.1")
(booster-set-param! booster "verbosity" "0")

(define eval-set (list (cons "train" dtrain) (cons "eval" deval)))
(define rounds 30)

(define (col s) (~a s #:width 10 #:align 'right))
(define (fmt v) (~r v #:precision '(= 4) #:min-width 10))

(printf "~a  ~a  ~a\n"
        (~a "iter" #:width 4)
        (col "train-rmse") (col "eval-rmse"))

(for ([iter (in-range rounds)])
  (booster-update-one-iter! booster iter dtrain)
  (define metrics (parse-eval-line (booster-eval-one-iter booster iter eval-set)))
  (printf "~a  ~a  ~a\n"
          (~a iter #:width 4)
          (fmt (hash-ref metrics "train-rmse"))
          (fmt (hash-ref metrics "eval-rmse"))))

(define final
  (parse-eval-line (booster-eval-one-iter booster (- rounds 1) eval-set)))
(printf "\nfinal train-rmse: ~a\n"
        (~r (hash-ref final "train-rmse") #:precision '(= 6)))
(printf "final  eval-rmse: ~a\n"
        (~r (hash-ref final "eval-rmse")  #:precision '(= 6)))

(booster-free! booster)
(dmatrix-free! dtrain)
(dmatrix-free! deval)
