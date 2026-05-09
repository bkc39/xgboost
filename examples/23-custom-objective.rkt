#lang racket/base

(require ffi/vector
         xgboost/ffi)

(provide run-example)

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

(define (mse preds)
  (/ (for/sum ([i (in-range (f32vector-length labels))])
       (define d (- (f32vector-ref preds i) (f32vector-ref labels i)))
       (* d d))
     (f32vector-length labels)))

(define (make-gradients preds)
  (define n (f32vector-length preds))
  (define grad (make-f32vector n))
  (define hess (make-f32vector n 1.0))
  (for ([i (in-range n)])
    ;; Squared-error objective: grad = prediction - label, hess = 1.
    (f32vector-set! grad i (- (f32vector-ref preds i)
                              (f32vector-ref labels i))))
  (values grad hess))

(define (run-example)
  (define dtrain (dmatrix-create-from-mat features 8 3 -1.0))
  (dmatrix-set-float-info! dtrain "label" labels)
  (define booster (booster-create (list dtrain)))
  (booster-set-param! booster "max_depth" "3")
  (booster-set-param! booster "eta" "0.2")
  (booster-set-param! booster "verbosity" "0")

  (define initial-preds (booster-predict booster dtrain #:output 'margin))
  (define initial-mse (mse initial-preds))

  (for ([iter (in-range 20)])
    (define preds (booster-predict booster dtrain #:output 'margin))
    (define-values (grad hess) (make-gradients preds))
    (booster-train-one-iter! booster iter dtrain grad hess))

  (define final-preds (booster-predict booster dtrain))
  (define final-mse (mse final-preds))
  (define result
    (hash 'prediction-count (f32vector-length final-preds)
          'initial-mse initial-mse
          'final-mse final-mse
          'improved? (< final-mse initial-mse)))

  (booster-free! booster)
  (dmatrix-free! dtrain)
  result)

(module+ main
  (define result (run-example))
  (printf "custom objective predictions: ~a\n"
          (hash-ref result 'prediction-count))
  (printf "initial MSE: ~a\n" (hash-ref result 'initial-mse))
  (printf "final MSE: ~a\n" (hash-ref result 'final-mse)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true (hash-ref result 'improved?))
  (check-true (< (hash-ref result 'final-mse) 3.0)))
