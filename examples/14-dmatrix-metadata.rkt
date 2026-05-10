#lang racket/base

(require ffi/vector
         racket/list
         xgboost)

(provide run-example)

(define (run-example)
  (define dm
    (make-dmatrix
     (f32vector 1.0 2.0
                3.0 4.0)
     #:nrow 2
     #:ncol 2
     #:missing -1.0))

  (dmatrix-set-feature-names! dm '("height" "weight"))
  (dmatrix-set-feature-types! dm '("q" "q"))
  (dmatrix-set-group! dm '(2))
  (dmatrix-set-label! dm (f32vector 0.25 0.75))

  (hash 'feature-names (dmatrix-feature-names dm)
        'feature-types (dmatrix-feature-types dm)
        'group-ptr (dmatrix-group-ptr dm)
        'labels (dmatrix-label dm)))

(module+ main
  (define result (run-example))
  (printf "feature names: ~a\n" (hash-ref result 'feature-names))
  (printf "feature types: ~a\n" (hash-ref result 'feature-types))
  (printf "group ptr: ~a\n" (hash-ref result 'group-ptr))
  (printf "labels: ~a\n" (hash-ref result 'labels)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-equal? (hash-ref result 'feature-names) '("height" "weight"))
  (check-equal? (hash-ref result 'feature-types) '("q" "q"))
  (check-equal? (hash-ref result 'group-ptr) '(0 2))
  (check-equal? (hash-ref result 'labels) '(0.25 0.75)))
