#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-print-dmatrix"]{Building a DMatrix}

The smallest interesting thing you can do with the bindings is build a
@tech{DMatrix} --- XGBoost's container for a feature matrix plus its
per-row metadata --- and read it back. Everything else (training, prediction)
takes a DMatrix as input, so this is the foundation.

We hand @racket[make-dmatrix] a flat @racket[f32vector] of feature values in
row-major order, tell it the shape with @racket[#:nrow]/@racket[#:ncol], and
attach a @racket[#:labels] vector (the regression target) and an optional
@racket[#:weights] vector (per-row importance). The underlying native handle is
reclaimed by Racket's garbage collector when the value becomes unreachable, so
there is nothing to free by hand.

@chunk[<r00-require>
(require ffi/vector
         xgboost)]

Every example in this chapter exports a @racket[run-example] thunk so the test
harness in @filepath{examples/test/} and this documentation can drive the same
code.

@chunk[<r00-provide>
(provide run-example)]

@bold{The data.} Four rows of three features each, laid out row by row. The
labels are the regression target; the weights up-weight the third row:

@chunk[<r00-data>
  (define features
    (f32vector 1.0 2.0 0.5
               2.0 1.0 1.5
               3.0 0.5 0.0
               0.5 3.0 2.0))
  (define labels  (f32vector 3.5 3.5 6.5 2.0))
  (define weights (f32vector 1.0 1.0 2.0 0.5))]

@bold{The matrix.} @racket[make-dmatrix] copies the data into a native DMatrix:

@chunk[<r00-build>
  (make-dmatrix features
                #:nrow 4
                #:ncol 3
                #:labels labels
                #:weights weights)]

@racket[run-example] returns that DMatrix. The companion harness
@filepath{test/00-print-dmatrix.rkt} renders it with @racket[dmatrix-show] and
materializes the rows with @racket[dmatrix->list] (missing entries would show as
@racket[+nan.0]); its @racket[test] submodule checks the shape and labels (run
with @exec{raco test}). You should see a 4×3 table whose labels read back as:

@racketblock[
(dmatrix-label (run-example))   (code:comment "'(3.5 3.5 6.5 2.0)")
]

@chunk[<r00-run>
(define (run-example)
  <r00-data>
  <r00-build>)]

@chunk[<*>
  <r00-require>
  <r00-provide>
  <r00-run>]
