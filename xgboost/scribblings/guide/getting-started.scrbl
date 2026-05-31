#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "getting-started"]{Getting Started}

This chapter is the Racket counterpart of XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/stable/get_started.html"]{Get
Started} quickstart. We train a classifier on the classic
@hyperlink["https://archive.ics.uci.edu/ml/datasets/iris"]{iris} dataset,
following the upstream Python example step for step.

@section{Classifying irises}

Here is the upstream Python quickstart:

@verbatim|{
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
}|

And the Racket translation:

@racketblock[
(require xgboost xgboost/private/demo-utils)

(code:comment "read data")
(define-values (X y) (load-iris))
(define-values (X-train X-test y-train y-test)
  (train-test-split X y #:test-size 0.2))

(code:comment "create model instance and fit")
(define bst
  (train (make-dmatrix X-train #:labels y-train)
         #:num-class 3
         #:objective "multi:softmax"
         #:max-depth 2
         #:eta 1.0
         #:rounds 2))

(code:comment "make predictions")
(define preds (predict bst (make-dmatrix X-test)))
]

The flow is the same; three details differ from the Python.

@itemlist[
 @item{@bold{There is no @tt{XGBClassifier}.} This binding exposes a single
       training interface: @racket[train] over a @tech{DMatrix}, then
       @racket[predict]. The estimator's @tt{n_estimators}, @tt{max_depth}, and
       @tt{learning_rate} map to @racket[#:rounds], @racket[#:max-depth], and
       @racket[#:eta].}
 @item{@bold{@racket[load-iris] and @racket[train-test-split] are demo helpers},
       not part of the library — they stand in for scikit-learn's
       @tt{load_iris} and @tt{train_test_split}. They live in
       @racketmodfont{xgboost/private/demo-utils}: @racket[load-iris] downloads
       the UCI iris CSV (falling back to a bundled copy offline) and returns
       @racket[(values X y)] — a list of feature rows and a list of integer
       labels — and @racket[train-test-split] returns the same four pieces as
       its Python namesake.}
 @item{@bold{iris has three classes,} so the objective is
       @racket["multi:softmax"] with @racket[#:num-class 3] rather than the
       @racket["binary:logistic"] of the two-class quickstart.
       @tt{XGBClassifier} picks the multiclass objective automatically; here we
       name it. With @racket["multi:softmax"], @racket[predict] returns the
       predicted class index per row, just as scikit-learn's @tt{.predict}
       returns labels.}
]

A complete, runnable, assertion-backed version is
@filepath{examples/27-get-started.rkt} in the package source.

@subsection{The demo helpers}

@defmodule[xgboost/private/demo-utils]

These helpers are bundled with the package purely to support the examples and
this guide — they stand in for the scikit-learn conveniences the upstream
quickstart relies on, and are not part of the modeling API.

@defproc[(load-iris)
         (values (listof (listof real?))
                 (listof exact-nonnegative-integer?))]{
Returns the iris dataset as @racket[(values X y)], where @racket[X] is a list of
four-element feature rows and @racket[y] is a list of integer class labels
(@racket[0], @racket[1], or @racket[2]). The data is downloaded from the UCI
repository; if the network is unavailable, a copy bundled with the package is
used instead, so the examples run offline.
}

@defproc[(train-test-split [X (listof (listof real?))]
                           [y (listof exact-nonnegative-integer?)]
                           [#:test-size test-size (real-in 0 1) 0.2]
                           [#:seed seed exact-integer? 0])
         (values (listof (listof real?))
                 (listof (listof real?))
                 (listof exact-nonnegative-integer?)
                 (listof exact-nonnegative-integer?))]{
Shuffles and partitions @racket[X] and @racket[y] into training and test sets,
returning @racket[(values X-train X-test y-train y-test)] like its scikit-learn
namesake. @racket[#:test-size] is the held-out fraction; @racket[#:seed] makes
the split reproducible.
}

@section{From in-memory Racket data}

The iris example builds its DMatrices from lists that @racket[load-iris]
returned. More generally, @racket[make-dmatrix] accepts a list of row lists, a
vector of row vectors, or a flat row-major sequence, with labels attached via
@racket[#:labels]. Here is a small regression instead of a classifier:

@racketblock[
(require xgboost)

(define dtrain
  (make-dmatrix '((1.0 2.0 0.5)
                  (2.0 1.0 1.5)
                  (3.0 0.5 0.0)
                  (0.5 3.0 2.0))
                #:labels '(3.5 3.5 6.5 2.0)))

(define booster
  (train dtrain
         #:objective "reg:squarederror"
         #:max-depth 2
         #:eta 0.2
         #:verbosity 0
         #:rounds 10))

(predict booster dtrain)
]

The most common parameters — @racket[#:objective], @racket[#:eta],
@racket[#:max-depth], and friends — have dedicated keywords as a shorthand for
entries in @racket[#:params]; @secref["parameters"] covers both styles. To read
data from a file instead, see @secref["data"]; to save and reload a trained
model, see @secref["training"].

@section{Where to go next}

The rest of this guide breaks the workflow down step by step:

@itemlist[
 @item{@secref["data"] — building and inspecting DMatrices, including sparse,
       columnar, and file inputs.}
 @item{@secref["parameters"] — the @racket[#:params] bundle and the keyword
       conveniences.}
 @item{@secref["training"] — fitting boosters, model IO, and driving the
       boosting loop for early stopping.}
 @item{@secref["prediction"] — output types and in-place prediction for
       serving.}
 @item{@secref["ranking"] — learning to rank with query groups.}
 @item{@secref["objectives"] — supplying a custom Racket objective.}
 @item{@secref["recipes"] — DART, monotonic constraints, and other parameter
       recipes.}
 @item{@secref["inspection"] — tree dumps and feature importance.}
 @item{@secref["config"] — global configuration, snapshots, and GPU training.}
]

For the precise contract of every procedure, see the @secref["reference"].
