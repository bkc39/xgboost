#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-dmatrix-metadata"]{DMatrix metadata}

A DMatrix carries more than features: you can attach and read back feature names
and types, ranking group boundaries, and labels. This example sets each on a
small matrix and round-trips them. Note @racket[dmatrix-set-group!] takes group
@emph{sizes} but @racket[dmatrix-group-ptr] returns the cumulative @emph{offsets}
(so a single group of 2 reads back as @racket['(0 2)]).

@chunk[<r14-require>
(require ffi/vector
         xgboost)]

@chunk[<r14-provide>
(provide run-example)]

@chunk[<r14-run>
(define (run-example)
  (define dm
    (make-dmatrix (f32vector 1.0 2.0  3.0 4.0)
                  #:nrow 2 #:ncol 2 #:missing -1.0))
  (dmatrix-set-feature-names! dm '("height" "weight"))
  (dmatrix-set-feature-types! dm '("q" "q"))
  (dmatrix-set-group! dm '(2))
  (dmatrix-set-label! dm (f32vector 0.25 0.75))
  (hash 'feature-names (dmatrix-feature-names dm)
        'feature-types (dmatrix-feature-types dm)
        'group-ptr (dmatrix-group-ptr dm)
        'labels (dmatrix-label dm)))]

The harness @filepath{test/14-dmatrix-metadata.rkt} prints each field and
asserts it round-trips.

@chunk[<*>
  <r14-require>
  <r14-provide>
  <r14-run>]
