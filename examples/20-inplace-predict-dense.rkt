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

(define (f32vector~= a b)
  (and (= (f32vector-length a) (f32vector-length b))
       (for/and ([i (in-range (f32vector-length a))])
         (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-6))))

(define (make-trained)
  (define dtrain (dmatrix-create-from-mat features 8 3 -1.0))
  (dmatrix-set-float-info! dtrain "label" labels)
  (define booster (booster-create (list dtrain)))
  (booster-set-param! booster "objective" "reg:squarederror")
  (booster-set-param! booster "max_depth" "3")
  (booster-set-param! booster "eta" "0.1")
  (booster-set-param! booster "verbosity" "0")
  (for ([iter (in-range 20)])
    (booster-update-one-iter! booster iter dtrain))
  (values booster dtrain))

(define (run-example)
  (define-values (booster dtrain) (make-trained))
  (define dmatrix-preds (booster-predict booster dtrain))
  (define inplace-preds
    (booster-predict-from-dense booster features 8 3 #:missing -1.0))
  (define result
    (hash 'prediction-count (f32vector-length inplace-preds)
          'matches-dmatrix? (f32vector~= inplace-preds dmatrix-preds)))
  (booster-free! booster)
  (dmatrix-free! dtrain)
  result)

(module+ main
  (define result (run-example))
  (printf "dense inplace predictions: ~a\n"
          (hash-ref result 'prediction-count))
  (printf "matches DMatrix prediction: ~a\n"
          (hash-ref result 'matches-dmatrix?)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true (hash-ref result 'matches-dmatrix?)))
