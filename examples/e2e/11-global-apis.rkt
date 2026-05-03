#lang racket/base

(require json
         racket/port
         xgboost)

(provide run-example)

(define (json-string->jsexpr s)
  (with-input-from-string s read-json))

(define (json-object-string? s)
  (hash? (json-string->jsexpr s)))

(define (run-example)
  (define build-info (xgboost-build-info))
  (define before-config (xgboost-get-global-config))
  (define during-config #f)

  (dynamic-wind
    void
    (lambda ()
      (xgboost-set-global-config! "{\"verbosity\":0}")
      (set! during-config (xgboost-get-global-config))
      (xgboost-register-log-callback! (lambda (_msg) (void))))
    (lambda ()
      (xgboost-set-global-config! before-config)))

  (define after-config (xgboost-get-global-config))
  (hash 'build-info build-info
        'before-config before-config
        'during-config during-config
        'after-config after-config))

(module+ main
  (define result (run-example))
  (printf "build info JSON: ~a\n" (json-object-string? (hash-ref result 'build-info)))
  (printf "verbosity set during example: ~a\n"
          (hash-ref (json-string->jsexpr (hash-ref result 'during-config))
                    'verbosity))
  (printf "global config restored: ~a\n"
          (equal? (json-string->jsexpr (hash-ref result 'before-config))
                  (json-string->jsexpr (hash-ref result 'after-config)))))

(module+ test
  (require rackunit)

  (define result (run-example))
  (check-true (json-object-string? (hash-ref result 'build-info)))
  (check-true (json-object-string? (hash-ref result 'before-config)))
  (check-true (json-object-string? (hash-ref result 'during-config)))
  (check-true (json-object-string? (hash-ref result 'after-config)))
  (check-equal? (hash-ref (json-string->jsexpr (hash-ref result 'during-config))
                          'verbosity)
                0)
  (check-equal? (json-string->jsexpr (hash-ref result 'after-config))
                (json-string->jsexpr (hash-ref result 'before-config))))
