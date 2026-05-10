#lang racket/base

(require ffi/vector
         json
         racket/port
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

(define (json-object? s)
  (hash? (with-input-from-string s read-json)))

(define (make-trained-booster rounds)
  (define dm
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (values (train dm
                 #:objective "reg:squarederror"
                 #:max-depth 3
                 #:eta 0.1
                 #:verbosity 0
                 #:rounds rounds)
          dm))

(define (run-example)
  (define-values (booster dm) (make-trained-booster 8))
  (define sliced (booster-slice booster 0 3))
  (define config (booster-config booster))
  (define loaded-config-booster (make-booster))
  (booster-set-config! loaded-config-booster config)
  (booster-reset! booster)
  (hash 'rounds (booster-boosted-rounds booster)
        'features (booster-num-feature booster)
        'sliced-rounds (booster-boosted-rounds sliced)
        'config-json? (json-object? config)))

(module+ main
  (define result (run-example))
  (printf "boosted rounds: ~a\n" (hash-ref result 'rounds))
  (printf "num features: ~a\n" (hash-ref result 'features))
  (printf "sliced rounds: ~a\n" (hash-ref result 'sliced-rounds))
  (printf "config JSON: ~a\n" (hash-ref result 'config-json?)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-equal? (hash-ref result 'rounds) 8)
  (check-equal? (hash-ref result 'features) 3)
  (check-equal? (hash-ref result 'sliced-rounds) 3)
  (check-true (hash-ref result 'config-json?)))
