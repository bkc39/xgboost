#lang racket/base

(require ffi/unsafe
         ffi/vector
         racket/list
         racket/string
         xgboost/ffi)

(provide run-example)

(define (array-interface-json ptr typestr dims)
  (format "{\"data\":[~a,false],\"typestr\":\"~a\",\"shape\":[~a],\"version\":3}"
          (cast ptr _pointer _uintptr)
          typestr
          (string-join (map number->string dims) ",")))

(define (f32-array-interface vec dims)
  (array-interface-json (f32vector->cpointer vec) "<f4" dims))

(define (u32vector->list vec)
  (for/list ([i (in-range (u32vector-length vec))])
    (u32vector-ref vec i)))

(define (f32vector->plain-list vec)
  (for/list ([i (in-range (f32vector-length vec))])
    (f32vector-ref vec i)))

(define (run-example)
  (define dm
    (dmatrix-create-from-dense
     (f32vector 1.0 2.0
                3.0 4.0)
     2
     2
     -1.0))
  (define labels (f32vector 0.25 0.75))

  (dmatrix-set-feature-info! dm "feature_name" '("height" "weight"))
  (dmatrix-set-feature-info! dm "feature_type" '("q" "q"))
  (dmatrix-set-uint-info! dm "group" (u32vector 2))
  (dmatrix-set-info-from-interface! dm "label"
                                    (f32-array-interface labels '(2)))

  (define result
    (hash 'feature-names (dmatrix-get-feature-info dm "feature_name")
          'feature-types (dmatrix-get-feature-info dm "feature_type")
          'group-ptr (u32vector->list (dmatrix-get-uint-info dm "group_ptr"))
          'labels (f32vector->plain-list (dmatrix-get-float-info dm "label"))))
  (dmatrix-free! dm)
  result)

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
