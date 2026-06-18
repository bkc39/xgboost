#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in polars dataframe?)
                     "../../main.rkt"))

@title[#:tag "getting-started"]{Getting Started}

This chapter trains a classifier on the classic
@hyperlink["https://archive.ics.uci.edu/ml/datasets/iris"]{iris} dataset,
mirroring XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/release_3.2.0/get_started.html"]{Python
Getting Started}. We read the data with @racketmodname[polars] and train
straight off the @tech{DataFrame}: @racket[make-dmatrix], @racket[train], and
@racket[predict] accept a Polars @racket[dataframe?] anywhere a @tech{DMatrix} is
expected (see @racket[dataframe->dmatrix]).

@racketblock[
(require xgboost
         (only-in polars read-csv ref len select series (filter df-filter)))

(code:comment "read data with polars; keep the four numeric feature columns")
(define features
  '("sepal_length" "sepal_width" "petal_length" "petal_width"))
(define df (read-csv "iris.csv"))
(define feature-df (select df features))

(code:comment "encode the string species column as integer classes 0/1/2")
(define species->class
  (hash "Iris-setosa" 0 "Iris-versicolor" 1 "Iris-virginica" 2))
(define labels
  (for/list ([i (in-range (len df))])
    (hash-ref species->class (ref (ref df "species") i))))

(code:comment "train directly on the DataFrame; #:labels supplies the targets")
(define bst
  (train feature-df
         #:labels labels
         #:num-class 3
         #:objective "multi:softprob"
         #:max-depth 3
         #:eta 0.3
         #:rounds 20))

(code:comment "predict on a (label-free) feature DataFrame")
(define probs (predict bst feature-df))
]

The upstream Python snippet passes @racket["binary:logistic"] to an
@tt{XGBClassifier} on this three-class problem --- a quirk that works only
because scikit-learn relabels internally. We use @racket["multi:softprob"] with
@racket[#:num-class] @racket[3] instead, which returns a per-row probability
vector (a flat list of @racket[length] @tt{3·N}) you reduce with @racket[argmax]
to a class.

A complete, runnable, assertion-backed version --- including a reproducible
train/test split and a Parquet round-trip --- is the @secref["ex-iris-polars"]
example. For the demo-helper path that downloads the dataset and uses an explicit
DMatrix instead, see the @secref["ex-get-started"] example.
