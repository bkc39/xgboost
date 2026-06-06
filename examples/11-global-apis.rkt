#lang scribble/lp2

@(require (for-label racket/base
                     xgboost))

@section[#:tag "ex-global-apis"]{Global and process APIs}

A few APIs are process-global rather than per-model: @racket[xgboost-build-info]
reports how the native library was compiled, and
@racket[xgboost-get-global-config] / @racket[xgboost-set-global-config!] read and
write process-wide settings as JSON. This example reads the build info, flips the
global verbosity inside a @racket[dynamic-wind] (restoring it afterward), and
registers a no-op log callback with @racket[xgboost-register-log-callback!].

@chunk[<r11-require>
(require xgboost)]

@chunk[<r11-provide>
(provide run-example)]

@chunk[<r11-run>
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
    (lambda () (xgboost-set-global-config! before-config)))
  (hash 'build-info build-info
        'before-config before-config
        'during-config during-config
        'after-config (xgboost-get-global-config)))]

The harness @filepath{test/11-global-apis.rkt} parses each returned string and
asserts the build info and configs are JSON objects, the verbosity took effect,
and the global config was restored.

@chunk[<*>
  <r11-require>
  <r11-provide>
  <r11-run>]
