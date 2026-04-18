#lang racket/base

(require racket/contract
         "ffi-raw.rkt")

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [run-regression-demo (-> rational?)]
  [run-classification-demo (-> (and/c rational? (>=/c 0) (<=/c 1)))]))

(define (xgboost-version)
  (xgb-version/raw))

(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (xgb-last-error/raw))))

(define (run-regression-demo)
  (define-values (rc out) (xgb-run-regression-demo/raw))
  (check-ok rc 'run-regression-demo)
  out)

(define (run-classification-demo)
  (define-values (rc out) (xgb-run-classification-demo/raw))
  (check-ok rc 'run-classification-demo)
  out)

(module+ test
  (require rackunit)
  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))
  (check-pred rational? (run-regression-demo))
  (define p (run-classification-demo))
  (check-true (<= 0 p 1)))
