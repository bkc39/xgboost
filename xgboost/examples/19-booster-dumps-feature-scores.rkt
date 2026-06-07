#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-booster-dumps"]{Model dumps and feature importance}

A trained model can be dumped as text or JSON, optionally with feature names,
and its feature importances read out. This example attaches names/types to both
the @tech{DMatrix} and the booster, then exercises @racket[booster-dump] (text,
JSON, and name-annotated) and @racket[booster-feature-score] (importance by
@racket["weight"]).

@chunk[<r19-require>
(require ffi/vector
         xgboost)]

@chunk[<r19-provide>
(provide run-example)]

@bold{Helpers.} The usual data plus shared feature metadata, and an
@racket[f32vector]→list converter:

@chunk[<r19-helpers>
  (define features
    (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
               4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define feature-names '("x0" "x1" "x2"))
  (define feature-types '("q" "q" "q"))
  (define (f32vector->plain-list vec)
    (for/list ([i (in-range (f32vector-length vec))]) (f32vector-ref vec i)))]

@chunk[<r19-run>
(define (run-example)
  (define dm (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  (dmatrix-set-feature-names! dm feature-names)
  (dmatrix-set-feature-types! dm feature-types)
  (define booster
    (train dm #:objective "reg:squarederror"
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 20))
  (booster-set-feature-names! booster feature-names)
  (booster-set-feature-types! booster feature-types)
  (define json-dumps (booster-dump booster #:format "json"))
  (define named-dumps
    (booster-dump booster #:format "text"
                  #:feature-names feature-names #:feature-types feature-types))
  (define scores
    (booster-feature-score booster #:importance-type "weight"
                           #:feature-names feature-names))
  (hash 'feature-info (booster-feature-names booster)
        'text-dump-count (length (booster-dump booster #:format "text"))
        'json-dump-has-object? (regexp-match? #rx"\\{" (car json-dumps))
        'named-dump-mentions-feature?
        (ormap (lambda (d) (regexp-match? #rx"x[0-2]" d)) named-dumps)
        'score-features (hash-ref scores 'features)
        'score-shape (hash-ref scores 'shape)
        'score-values (f32vector->plain-list (hash-ref scores 'scores))))]

The harness @filepath{test/19-booster-dumps-feature-scores.rkt} prints the
dump count and importances, and asserts the dumps and feature scores are
well-formed.

@chunk[<*>
  <r19-require>
  <r19-provide>
  <r19-helpers>
  <r19-run>]
