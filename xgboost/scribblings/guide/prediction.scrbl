#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "prediction"]{Prediction}

@racket[predict] runs a trained booster over a DMatrix. By default it returns a
list of predictions; pass @racket[#:as 'f32vector] to get the raw
@racket[f32vector?] copied from XGBoost instead:

@racketblock[
(require xgboost)

(define preds (predict booster dtrain))                 (code:comment "list of reals")
(define vec   (predict booster dtrain #:as 'f32vector)) (code:comment "f32vector")
]

For a @racket["binary:logistic"] model the values are class-1 probabilities; for
@racket["multi:softprob"] the output has @racket[nrow]×@racket[num-class] entries
(row-major), while @racket["multi:softmax"] yields one predicted class index per
row.

@section{Choosing what to predict}

@racket[#:output] selects the kind of output. @racket['value] (the default) is
the final prediction; @racket['margin] is the raw score before the logistic or
softmax transform; @racket['leaf] gives per-tree leaf indices; and
@racket['contribs] / @racket['approx-contribs] /
@racket['interactions] / @racket['approx-interactions] return SHAP-style feature
attributions:

@racketblock[
(predict booster dtrain #:output 'margin)
(predict booster dtrain #:output 'contribs)
]

@racket[#:iteration-end] limits prediction to the first @racket[N] boosting
rounds (@racket[0], the default, means all of them) — useful for evaluating an
early-stopping prefix without re-slicing the booster.

@section{In-place prediction}

For serving, you often have a single batch of raw features and no reason to
build a DMatrix first. The @racket[predict-from-dense],
@racket[predict-from-csr], and @racket[predict-from-columnar] variants predict
directly from Racket data and accept the same @racket[#:output],
@racket[#:iteration-end], and @racket[#:as] keywords as @racket[predict]:

@racketblock[
(code:comment "dense rows, shaped like make-dmatrix's data argument")
(predict-from-dense booster features
                    #:nrow 8 #:ncol 3
                    #:missing -1.0
                    #:as 'f32vector)

(code:comment "CSR triple: indptr, indices, values, ncol")
(predict-from-csr booster indptr indices values 3
                  #:missing -1.0)

(code:comment "one f32vector per column")
(predict-from-columnar booster
                       (list (f32vector 1.0 2.0 3.0)
                             (f32vector 2.0 1.0 0.5)
                             (f32vector 0.5 1.5 0.0))
                       #:missing -1.0)
]

In-place prediction returns the same numbers as building a DMatrix and calling
@racket[predict]; it just skips the intermediate allocation.
