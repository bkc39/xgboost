#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-booster-lifecycle"]{Booster lifecycle and config}

A trained booster can be queried, sliced, reset, and round-tripped through its
JSON config. This example trains eight rounds, then exercises
@racket[booster-slice] (take a contiguous range of rounds into a new booster),
@racket[booster-config] / @racket[booster-set-config!] (serialize and restore
the full configuration as JSON), @racket[booster-reset!] (drop the boosted
trees), and the @racket[booster-boosted-rounds] / @racket[booster-num-feature]
queries.

@chunk[<r17-require>
(require ffi/vector
         json
         racket/port
         xgboost)]

@chunk[<r17-provide>
(provide run-example)]

@bold{Helpers.} A trained booster and a JSON-object check:

@chunk[<r17-helpers>
  (define features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define (json-object? s) (hash? (with-input-from-string s read-json)))
  (define (make-trained-booster rounds)
    (define dm (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
    (values (train dm #:objective "reg:squarederror"
                   #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds rounds)
            dm))]

@bold{The run.} @racket[run-example] returns the post-reset round count, the
feature count, the sliced booster's round count, and whether the config
serializes to a JSON object:

@chunk[<r17-run>
(define (run-example)
  (define-values (booster dm) (make-trained-booster 8))
  (define sliced (booster-slice booster 0 3))
  (define config (booster-config booster))
  (booster-set-config! (make-booster) config)
  (booster-reset! booster)
  (hash 'rounds (booster-boosted-rounds booster)
        'features (booster-num-feature booster)
        'sliced-rounds (booster-boosted-rounds sliced)
        'config-json? (json-object? config)))]

The harness @filepath{test/17-booster-lifecycle-config.rkt} prints these and
asserts the slice kept 3 rounds, the matrix has 3 features, and the config is
valid JSON.

@chunk[<*>
  <r17-require>
  <r17-provide>
  <r17-helpers>
  <r17-run>]
