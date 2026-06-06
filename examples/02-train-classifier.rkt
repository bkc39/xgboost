#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-train-classifier"]{Binary classification}

Switching from regression to classification is a one-parameter change: the
objective. This example fits @racket["binary:logistic"], whose predictions are
@emph{probabilities} in @tt{[0, 1]} rather than raw targets. We threshold at
@tt{0.5} to recover a hard class and report accuracy.

The data is two well-separated clusters in 4-D space: class 0 rows are
small-valued, class 1 rows are large-valued, interleaved so the labels
alternate.

@chunk[<r02-require>
(require ffi/vector
         xgboost)]

@chunk[<r02-provide>
(provide run-example)]

@bold{The data.} Ten rows of four features, alternating class 0 / class 1:

@chunk[<r02-data>
  (define features
    (f32vector 0.1 0.2 0.1 0.0
               5.0 4.0 5.5 6.0
               0.3 0.5 0.1 0.2
               4.5 5.0 4.0 5.5
               0.0 0.1 0.2 0.0
               6.0 5.5 6.5 5.0
               0.4 0.3 0.2 0.5
               5.5 6.0 4.5 5.0
               0.2 0.1 0.3 0.1
               4.0 4.5 5.0 4.0))
  (define labels (f32vector 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))
  (define dtrain
    (make-dmatrix features #:nrow 10 #:ncol 4 #:labels labels))]

@bold{Training.} The only change from @secref["ex-train-regression"] is
@racket[#:objective], now @racket["binary:logistic"]:

@chunk[<r02-train>
  (define booster
    (train dtrain
           #:objective "binary:logistic"
           #:max-depth 3
           #:eta 0.3
           #:verbosity 0
           #:rounds 30))]

@bold{Prediction.} For @racket["binary:logistic"], @racket[predict] returns the
probability of class 1 for each row:

@chunk[<r02-predict>
  (define probs (predict booster dtrain #:as 'f32vector))]

@racket[run-example] returns the booster, the training matrix, and those
probabilities. The harness @filepath{test/02-train-classifier.rkt} prints a
truth/probability/prediction table, reports accuracy at the @tt{0.5} threshold,
and asserts the clusters are classified perfectly:

@racketblock[
(code:comment "i    truth     p(1)     pred")
(code:comment "0        0   0.0123      0  ")
(code:comment "1        1   0.9881      1  ")
(code:comment "accuracy: 10/10 (100.0%)")
]

@chunk[<r02-run>
(define (run-example)
  <r02-data>
  <r02-train>
  <r02-predict>
  (values booster dtrain probs))]

@chunk[<*>
  <r02-require>
  <r02-provide>
  <r02-run>]
