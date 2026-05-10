#lang racket/base

(require ffi/vector
         racket/list
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
(define feature-names '("x0" "x1" "x2"))
(define feature-types '("q" "q" "q"))

(define (f32vector->plain-list vec)
  (for/list ([i (in-range (f32vector-length vec))])
    (f32vector-ref vec i)))

(define (run-example)
  (define dm
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (dmatrix-set-feature-names! dm feature-names)
  (dmatrix-set-feature-types! dm feature-types)
  (define booster
    (train dm
           #:objective "reg:squarederror"
           #:max-depth 3
           #:eta 0.1
           #:verbosity 0
           #:rounds 20))
  (booster-set-feature-names! booster feature-names)
  (booster-set-feature-types! booster feature-types)

  (define text-dumps (booster-dump booster #:format "text"))
  (define json-dumps (booster-dump booster #:format "json"))
  (define named-dumps
    (booster-dump booster
                  #:format "text"
                  #:feature-names feature-names
                  #:feature-types feature-types))
  (define scores
    (booster-feature-score booster
                           #:importance-type "weight"
                           #:feature-names feature-names))
  (hash 'feature-info (booster-feature-names booster)
        'text-dump-count (length text-dumps)
        'json-dump-has-object? (regexp-match? #rx"\\{" (car json-dumps))
        'named-dump-mentions-feature?
        (ormap (lambda (dump) (regexp-match? #rx"x[0-2]" dump)) named-dumps)
        'score-features (hash-ref scores 'features)
        'score-shape (hash-ref scores 'shape)
        'score-values (f32vector->plain-list (hash-ref scores 'scores))))

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
