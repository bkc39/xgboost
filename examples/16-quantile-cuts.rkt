#lang racket/base

(require ffi/vector
         racket/list
         xgboost)

(provide run-example)

(define (run-example)
  (define dtrain
    (make-dmatrix
     (f32vector 1.0 2.0
                3.0 4.0
                5.0 6.0
                7.0 8.0)
     #:nrow 4
     #:ncol 2
     #:missing -1.0
     #:labels (f32vector 1.0 3.0 5.0 7.0)))
  (define booster
    (train dtrain
           #:objective "reg:squarederror"
           #:params '(("tree_method" . "hist"))
           #:max-depth 2
           #:verbosity 0
           #:rounds 1))
  (define-values (indptr data)
    (dmatrix-quantile-cut dtrain))
  (hash 'indptr indptr
        'data data
        'indptr-length (length indptr)
        'data-length (f32vector-length data)
        'prediction-count (f32vector-length
                           (predict booster dtrain #:as 'f32vector))))

(module+ main
  (define result (run-example))
  (printf "quantile indptr length: ~a\n" (hash-ref result 'indptr-length))
  (printf "quantile data length: ~a\n" (hash-ref result 'data-length))
  (printf "prediction count: ~a\n" (hash-ref result 'prediction-count)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (define indptr (hash-ref result 'indptr))
  (define data (hash-ref result 'data))
  (check-true (positive? (length indptr)))
  (check-equal? (first indptr) 0)
  (check-true (positive? (last indptr)))
  (check-true (f32vector? data))
  (check-equal? (f32vector-length data) (last indptr))
  (check-equal? (hash-ref result 'prediction-count) 4))
