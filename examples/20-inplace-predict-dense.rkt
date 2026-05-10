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

(define (f32vector~= a b)
  (and (= (f32vector-length a) (f32vector-length b))
       (for/and ([i (in-range (f32vector-length a))])
         (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-6))))

(define (make-trained)
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (values (train dtrain
                 #:objective "reg:squarederror"
                 #:max-depth 3
                 #:eta 0.1
                 #:verbosity 0
                 #:rounds 20)
          dtrain))

(define (run-example)
  (define-values (booster dtrain) (make-trained))
  (define dmatrix-preds (predict booster dtrain #:as 'f32vector))
  (define inplace-preds
    (predict-from-dense booster features
                        #:nrow 8 #:ncol 3
                        #:missing -1.0
                        #:as 'f32vector))
  (hash 'prediction-count (f32vector-length inplace-preds)
        'inplace-preds    inplace-preds
        'dmatrix-preds    dmatrix-preds
        'matches-dmatrix? (f32vector~= inplace-preds dmatrix-preds)))

(module+ main
  (define result (run-example))
  (printf "dense inplace predictions: ~a\n"
          (hash-ref result 'prediction-count))
  (printf "matches DMatrix prediction: ~a\n"
          (hash-ref result 'matches-dmatrix?)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (define inplace (hash-ref result 'inplace-preds))
  (define dmatrix (hash-ref result 'dmatrix-preds))
  (define n (f32vector-length inplace))
  (define diffs
    (for/list ([i (in-range n)])
      (abs (- (f32vector-ref inplace i) (f32vector-ref dmatrix i)))))
  (define max-diff (apply max diffs))
  (check-equal? (hash-ref result 'prediction-count) 8)
  (check-true
   (hash-ref result 'matches-dmatrix?)
   (format "inplace vs dmatrix predictions differ beyond 1e-4\n  inplace: ~a\n  dmatrix: ~a\n  per-element diffs: ~a\n  max diff: ~a"
           inplace dmatrix diffs max-diff)))
