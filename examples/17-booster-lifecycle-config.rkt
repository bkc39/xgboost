#lang racket/base

(require json
         ffi/vector
         racket/port
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

(define (json-object? s)
  (hash? (with-input-from-string s read-json)))

(define (make-trained-booster rounds)
  (define dm (dmatrix-create-from-mat features 8 3 -1.0))
  (dmatrix-set-float-info! dm "label" labels)
  (define booster (booster-create (list dm)))
  (booster-set-param! booster "objective" "reg:squarederror")
  (booster-set-param! booster "max_depth" "3")
  (booster-set-param! booster "eta" "0.1")
  (booster-set-param! booster "verbosity" "0")
  (for ([iter (in-range rounds)])
    (booster-update-one-iter! booster iter dm))
  (values booster dm))

(define (run-example)
  (define-values (booster dm) (make-trained-booster 8))
  (define sliced (booster-slice booster 0 3))
  (define config (booster-save-json-config booster))
  (define loaded-config-booster (booster-create))
  (booster-load-json-config! loaded-config-booster config)
  (booster-reset! booster)
  (define result
    (hash 'rounds (booster-boosted-rounds booster)
          'features (booster-num-feature booster)
          'sliced-rounds (booster-boosted-rounds sliced)
          'config-json? (json-object? config)))
  (booster-free! loaded-config-booster)
  (booster-free! sliced)
  (booster-free! booster)
  (dmatrix-free! dm)
  result)

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
