#lang racket/base

;; Booster persistence: model save/load to and from filesystem paths and
;; bytes, full-state serialize/unserialize snapshots, and JSON config
;; round-trips.

(require "../error.rkt"
         "../raw/booster.rkt")

(provide booster-save-model!
         booster-load-model!
         booster-save-model-to-bytes
         booster-load-model-from-bytes!
         booster-serialize-to-bytes
         booster-unserialize-from-bytes!
         booster-save-json-config
         booster-load-json-config!)

(define (booster-save-model! h path)
  (check-ok (xgb-booster-save-model/raw h path) 'booster-save-model!))

(define (booster-load-model! h path)
  (check-ok (xgb-booster-load-model/raw h path) 'booster-load-model!))

;; Serialize a booster to a fresh bytes object.  `format` is "ubj" (compact
;; binary, default) or "json" (human-readable).  Same size-then-fill dance
;; as predict: probe to get the required length, allocate, refill.
(define (booster-save-model-to-bytes h #:format [fmt "ubj"])
  (define config (string-append "{\"format\":\"" fmt "\"}"))
  (define-values (rc len)
    (xgb-booster-save-model-to-buffer/raw h config 0 (make-bytes 0)))
  (cond
    [(zero? rc) (make-bytes len)]   ; degenerate: 0-byte model
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-save-model-to-buffer/raw h config len buf))
     (check-ok rc2 'booster-save-model-to-bytes)
     (unless (= len2 len)
       (error 'booster-save-model-to-bytes
              "expected ~a bytes, got ~a" len len2))
     buf]
    [else (check-ok rc 'booster-save-model-to-bytes)]))

(define (booster-load-model-from-bytes! h buf)
  (check-ok (xgb-booster-load-model-from-buffer/raw h buf)
            'booster-load-model-from-bytes!))

;; Snapshot the full booster state (including training caches and iteration
;; counters), so an unserialized handle can keep calling
;; `booster-update-one-iter!` and produce the same trajectory as the
;; original.  Same size-then-fill probe as save-model-to-bytes.
(define (booster-serialize-to-bytes h)
  (define-values (rc len)
    (xgb-booster-serialize-to-buffer/raw h 0 (make-bytes 0)))
  (cond
    [(zero? rc) (make-bytes len)]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-serialize-to-buffer/raw h len buf))
     (check-ok rc2 'booster-serialize-to-bytes)
     (unless (= len2 len)
       (error 'booster-serialize-to-bytes
              "expected ~a bytes, got ~a" len len2))
     buf]
    [else (check-ok rc 'booster-serialize-to-bytes)]))

(define (booster-unserialize-from-bytes! h buf)
  (check-ok (xgb-booster-unserialize-from-buffer/raw h buf)
            'booster-unserialize-from-bytes!))

(define (booster-save-json-config h)
  (copy-string-result 'booster-save-json-config
                      (lambda (capacity buf)
                        (xgb-booster-save-json-config/raw h capacity buf))))

(define (booster-load-json-config! h config)
  (check-ok (xgb-booster-load-json-config/raw h config)
            'booster-load-json-config!))
