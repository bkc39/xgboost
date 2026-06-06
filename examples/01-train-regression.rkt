#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-train-regression"]{Training a regressor}

With a @tech{DMatrix} in hand, training is one call. This example fits a
gradient-boosted regression model end to end: build a small synthetic dataset,
@racket[train] a booster, @racket[predict] on the same rows, and compare the
predictions to the labels.

The labels were chosen to follow roughly @tt{2·x₀ + x₁ − x₂} with a touch of
noise --- a relationship a tree booster fits easily.

@chunk[<r01-require>
(require ffi/vector
         xgboost)]

@chunk[<r01-provide>
(provide run-example)]

@bold{The data.} Eight rows of three features, with the regression target in
@racket[labels]:

@chunk[<r01-data>
  (define features
    (f32vector 1.0 2.0 0.5
               2.0 1.0 1.5
               3.0 0.5 0.0
               0.5 3.0 2.0
               4.0 2.0 1.0
               1.5 1.5 0.5
               2.5 3.5 1.5
               0.0 1.0 0.0))
  (define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:labels labels))]

@bold{Training.} @racket[train] runs the boosting loop. The objective
@racket["reg:squarederror"] is ordinary least-squares regression; @racket[#:eta]
is the learning rate and @racket[#:max-depth] caps each tree. We keep
@racket[#:verbosity] at @racket[0] so the run is quiet:

@chunk[<r01-train>
  (define booster
    (train dtrain
           #:objective "reg:squarederror"
           #:max-depth 3
           #:eta 0.1
           #:verbosity 0
           #:rounds 50))]

@bold{Prediction.} @racket[predict] with @racket[#:as 'f32vector] returns the
per-row predictions as an @racket[f32vector]:

@chunk[<r01-predict>
  (define preds (predict booster dtrain #:as 'f32vector))]

@racket[run-example] returns the booster, the training matrix, and those
predictions. The companion harness @filepath{test/01-train-regression.rkt}
prints a predictions-vs-labels table, reports the training MSE, and its
@racket[test] submodule asserts the model fits (run with @exec{raco test}).
After 50 rounds the predictions track the labels closely:

@racketblock[
(code:comment "i    label      pred")
(code:comment "0   3.5000    3.5012")
(code:comment "...                 ")
(code:comment "training MSE ≈ 0.000…")
]

@chunk[<r01-run>
(define (run-example)
  <r01-data>
  <r01-train>
  <r01-predict>
  (values booster dtrain preds))]

@chunk[<*>
  <r01-require>
  <r01-provide>
  <r01-run>]
