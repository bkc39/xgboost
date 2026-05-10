#lang racket/base

(require ffi/vector
         xgboost)

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

;; Squared-error gradient: grad = prediction - label, hess = 1.
(define (squared-error preds _dtrain)
  (define n (f32vector-length preds))
  (define grad (make-f32vector n))
  (define hess (make-f32vector n 1.0))
  (for ([i (in-range n)])
    (f32vector-set! grad i (- (f32vector-ref preds i)
                              (f32vector-ref labels i))))
  (values grad hess))

(define (run-example)
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  ;; Zero-round booster bound to dtrain: same shape as the trained one,
  ;; just no boosting steps applied yet.  Margin defaults to base_score.
  (define baseline
    (train dtrain #:max-depth 3 #:eta 0.2 #:verbosity 0 #:rounds 0))
  (define initial-preds
    (predict baseline dtrain #:output 'margin #:as 'f32vector))
  (define initial-mse (mse initial-preds))

  (define booster
    (train dtrain
           #:objective-fn squared-error
           #:max-depth 3
           #:eta 0.2
           #:verbosity 0
           #:rounds 20))

  (define final-preds (predict booster dtrain #:as 'f32vector))
  (define final-mse (mse final-preds))
  (hash 'prediction-count (f32vector-length final-preds)
        'initial-mse initial-mse
        'final-mse final-mse
        'improved? (< final-mse initial-mse)))

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
