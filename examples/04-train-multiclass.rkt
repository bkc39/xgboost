#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     xgboost))

@section[#:tag "ex-train-multiclass"]{Multiclass classification}

Multiclass problems need two extra parameters --- @racket[#:num-class] and a
multiclass @racket[#:objective] --- and they come in two output flavors:

@itemlist[
  @item{@racket["multi:softprob"] emits @tt{nrow × num_class} probabilities; the
        per-row probabilities sum to 1, and the predicted class is the argmax.}
  @item{@racket["multi:softmax"] emits @tt{nrow} predicted class indices
        directly.}]

This example trains both on the same data: three tight clusters in 2-D, six
rows per class, centered at @tt{(0,0)}, @tt{(6,0)}, and @tt{(3,6)}.

@chunk[<r04-require>
(require ffi/vector
         xgboost)]

@chunk[<r04-provide>
(provide run-example)]

@bold{The data.} Eighteen rows, six per class:

@chunk[<r04-data>
  (define features
    (f32vector
     0.0  0.0    1.0  0.0    0.0  1.0    -1.0  0.0    0.0 -1.0    1.0  1.0
     6.0  0.0    7.0  0.0    6.0  1.0     5.0  0.0    6.0 -1.0    7.0  1.0
     3.0  6.0    4.0  6.0    3.0  7.0     2.0  6.0    3.0  5.0    4.0  7.0))
  (define labels (f32vector 0.0 0.0 0.0 0.0 0.0 0.0
                            1.0 1.0 1.0 1.0 1.0 1.0
                            2.0 2.0 2.0 2.0 2.0 2.0))
  (define dtrain
    (make-dmatrix features #:nrow 18 #:ncol 2 #:labels labels))]

@bold{Training both objectives.} A small helper trains the shared data under a
given objective; the @racket[#:num-class] is what makes it multiclass:

@chunk[<r04-train>
  (define (train-booster objective)
    (train dtrain
           #:objective objective
           #:num-class 3
           #:max-depth 3
           #:eta 0.3
           #:verbosity 0
           #:rounds 30))]

@bold{Prediction.} @racket["multi:softprob"] returns a flat @tt{18 × 3} block of
probabilities; @racket["multi:softmax"] returns 18 class indices:

@chunk[<r04-predict>
  (define probs (predict (train-booster "multi:softprob") dtrain #:as 'f32vector))
  (define preds (predict (train-booster "multi:softmax")  dtrain #:as 'f32vector))]

@racket[run-example] returns the labels, the softprob block, and the softmax
indices. The harness @filepath{test/04-train-multiclass.rkt} prints the
per-row probabilities with their argmax, checks that each row's probabilities
sum to 1, and asserts both objectives recover every label:

@racketblock[
(code:comment "softprob output: 54 floats (= nrow * nclass)")
(code:comment "softprob argmax accuracy: 18/18")
(code:comment "softmax  output: 18 floats (= nrow)")
(code:comment "softmax  accuracy: 18/18")
]

@chunk[<r04-run>
(define (run-example)
  <r04-data>
  <r04-train>
  <r04-predict>
  (values labels probs preds))]

@chunk[<*>
  <r04-require>
  <r04-provide>
  <r04-run>]
