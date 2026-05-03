#lang racket/base

(require ffi/vector
         racket/list
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
(define feature-names '("x0" "x1" "x2"))
(define feature-types '("q" "q" "q"))

(define (f32vector->plain-list vec)
  (for/list ([i (in-range (f32vector-length vec))])
    (f32vector-ref vec i)))

(define (run-example)
  (define dm (dmatrix-create-from-mat features 8 3 -1.0))
  (dmatrix-set-float-info! dm "label" labels)
  (dmatrix-set-feature-info! dm "feature_name" feature-names)
  (dmatrix-set-feature-info! dm "feature_type" feature-types)
  (define booster (booster-create (list dm)))
  (booster-set-param! booster "objective" "reg:squarederror")
  (booster-set-param! booster "max_depth" "3")
  (booster-set-param! booster "eta" "0.1")
  (booster-set-param! booster "verbosity" "0")
  (booster-set-feature-info! booster "feature_name" feature-names)
  (booster-set-feature-info! booster "feature_type" feature-types)
  (for ([iter (in-range 20)])
    (booster-update-one-iter! booster iter dm))

  (define text-dumps (booster-dump-model booster #:format "text"))
  (define json-dumps (booster-dump-model booster #:format "json"))
  (define named-dumps
    (booster-dump-model-with-features booster feature-names feature-types
                                      #:format "text"))
  (define scores
    (booster-feature-score booster
                           #:importance-type "weight"
                           #:feature-names feature-names))
  (define result
    (hash 'feature-info (booster-get-feature-info booster "feature_name")
          'text-dump-count (length text-dumps)
          'json-dump-has-object? (regexp-match? #rx"\\{" (car json-dumps))
          'named-dump-mentions-feature?
          (ormap (lambda (dump) (regexp-match? #rx"x[0-2]" dump)) named-dumps)
          'score-features (hash-ref scores 'features)
          'score-shape (hash-ref scores 'shape)
          'score-values (f32vector->plain-list (hash-ref scores 'scores))))
  (booster-free! booster)
  (dmatrix-free! dm)
  result)

(module+ main
  (define result (run-example))
  (printf "booster feature names: ~a\n" (hash-ref result 'feature-info))
  (printf "dump count: ~a\n" (hash-ref result 'text-dump-count))
  (printf "score features: ~a\n" (hash-ref result 'score-features))
  (printf "score values: ~a\n" (hash-ref result 'score-values)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-equal? (hash-ref result 'feature-info) feature-names)
  (check-true (> (hash-ref result 'text-dump-count) 0))
  (check-true (hash-ref result 'json-dump-has-object?))
  (check-true (hash-ref result 'named-dump-mentions-feature?))
  (check-true (pair? (hash-ref result 'score-features)))
  (check-true (pair? (hash-ref result 'score-shape)))
  (check-true (andmap positive? (hash-ref result 'score-values))))
