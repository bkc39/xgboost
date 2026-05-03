#lang racket/base

(require json
         ffi/vector
         racket/port
         xgboost/ffi)

(provide run-example)

(define (json-string->jsexpr s)
  (with-input-from-string s read-json))

(define (shape-of-array-interface s)
  (hash-ref (json-string->jsexpr s) 'shape))

(define (run-example)
  (define dtrain
    (dmatrix-create-from-dense
     (f32vector 1.0 2.0
                3.0 4.0
                5.0 6.0
                7.0 8.0)
     4
     2
     -1.0))
  (dmatrix-set-float-info! dtrain "label" (f32vector 1.0 3.0 5.0 7.0))
  (define booster (booster-create (list dtrain)))
  (booster-set-param! booster "objective" "reg:squarederror")
  (booster-set-param! booster "tree_method" "hist")
  (booster-set-param! booster "max_depth" "2")
  (booster-set-param! booster "verbosity" "0")
  (booster-update-one-iter! booster 0 dtrain)
  (define-values (indptr data)
    (dmatrix-get-quantile-cut dtrain))
  (define result
    (hash 'indptr indptr
          'data data
          'indptr-shape (shape-of-array-interface indptr)
          'data-shape (shape-of-array-interface data)
          'prediction-count (f32vector-length (booster-predict booster dtrain))))
  (booster-free! booster)
  (dmatrix-free! dtrain)
  result)

(module+ main
  (define result (run-example))
  (printf "quantile indptr shape: ~a\n" (hash-ref result 'indptr-shape))
  (printf "quantile data shape: ~a\n" (hash-ref result 'data-shape))
  (printf "prediction count: ~a\n" (hash-ref result 'prediction-count)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-regexp-match #rx"^\\{" (hash-ref result 'indptr))
  (check-regexp-match #rx"^\\{" (hash-ref result 'data))
  (check-true (pair? (hash-ref result 'indptr-shape)))
  (check-true (pair? (hash-ref result 'data-shape)))
  (check-true (> (car (hash-ref result 'indptr-shape)) 0))
  (check-true (> (car (hash-ref result 'data-shape)) 0))
  (check-equal? (hash-ref result 'prediction-count) 4))
