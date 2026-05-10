#lang racket/base

(require xgboost)

(provide run-example)

(define (run-example)
  (define b (make-booster))
  (booster-set-attr! b "owner" "racket")
  (booster-set-attr! b "purpose" "example")
  (define before-delete
    (hash 'owner (booster-attr b "owner")
          'purpose (booster-attr b "purpose")
          'names (sort (booster-attr-names b) string<?)))
  (booster-delete-attr! b "purpose")
  (hash 'before-delete before-delete
        'purpose-after-delete (booster-attr b "purpose")
        'names-after-delete (booster-attr-names b)))

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
