#lang racket/base

(require xgboost/ffi)

(provide run-example)

(define (run-example)
  (define booster (booster-create))
  (booster-set-attr! booster "owner" "racket")
  (booster-set-attr! booster "purpose" "example")
  (define before-delete
    (hash 'owner (booster-get-attr booster "owner")
          'purpose (booster-get-attr booster "purpose")
          'names (sort (booster-get-attr-names booster) string<?)))
  (booster-delete-attr! booster "purpose")
  (define result
    (hash 'before-delete before-delete
          'purpose-after-delete (booster-get-attr booster "purpose")
          'names-after-delete (booster-get-attr-names booster)))
  (booster-free! booster)
  result)

(module+ main
  (define result (run-example))
  (define before (hash-ref result 'before-delete))
  (printf "owner: ~a\n" (hash-ref before 'owner))
  (printf "attrs before delete: ~a\n" (hash-ref before 'names))
  (printf "purpose after delete: ~a\n"
          (hash-ref result 'purpose-after-delete)))

(module+ test
  (require rackunit)

  (define result (run-example))
  (define before (hash-ref result 'before-delete))
  (check-equal? (hash-ref before 'owner) "racket")
  (check-equal? (hash-ref before 'purpose) "example")
  (check-equal? (hash-ref before 'names) '("owner" "purpose"))
  (check-false (hash-ref result 'purpose-after-delete))
  (check-equal? (hash-ref result 'names-after-delete) '("owner")))
