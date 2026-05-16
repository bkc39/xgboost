#lang racket/base

;; Shared error-handling and result-copying helpers for the safe layer.
;;
;; Most XGBoost C functions follow one of two contracts:
;;   - return an integer status code (0 ok), with `xgb_last_error` holding
;;     the message — `check-ok` turns a non-zero code into a Racket error.
;;   - a "size-or-fill" probe: a first call with a zero-capacity buffer
;;     reports the required size via rc=2, then a second call fills it.
;;     `copy-string-result` / `copy-nul-separated-result` drive that dance.

(require "raw/global.rkt")

(provide check-ok
         check-handle
         copy-string-result
         copy-nul-separated-result
         nul-separated-bytes->strings)

(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (xgb-last-error/raw))))

(define (check-handle who h)
  (unless h
    (error who "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  h)

;; Drive a size-or-fill probe whose raw call has the shape
;;   (raw capacity buf) -> (values rc out-len)
;; and return the filled buffer decoded as a UTF-8 string.
(define (copy-string-result who raw)
  (define-values (rc len)
    (raw 0 (make-bytes 0)))
  (cond
    [(zero? rc) ""]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2) (raw len buf))
     (check-ok rc2 who)
     (unless (= len2 len)
       (error who "expected ~a bytes, got ~a" len len2))
     (bytes->string/utf-8 buf #f 0 len)]
    [else (check-ok rc who)]))

(define (nul-separated-bytes->strings bs count)
  (define len (bytes-length bs))
  (let loop ([start 0] [i 0] [acc '()])
    (cond
      [(= i count) (reverse acc)]
      [else
       (define end
         (let find-nul ([j start])
           (cond
             [(>= j len) len]
             [(zero? (bytes-ref bs j)) j]
             [else (find-nul (add1 j))])))
       (loop (add1 end)
             (add1 i)
             (cons (bytes->string/utf-8 bs #f start end) acc))])))

;; Size-or-fill probe whose raw call has the shape
;;   (raw capacity buf) -> (values rc out-len out-count)
;; returning a list of NUL-separated strings.
(define (copy-nul-separated-result who raw)
  (define-values (rc len count) (raw 0 (make-bytes 0)))
  (cond
    [(zero? rc) '()]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 count2) (raw len buf))
     (check-ok rc2 who)
     (unless (= len2 len)
       (error who "expected ~a bytes, got ~a" len len2))
     (nul-separated-bytes->strings buf count2)]
    [else (check-ok rc who)]))
