#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"))

@title[#:tag "getting-started"]{Getting Started}

This chapter is the Racket counterpart of XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/stable/get_started.html"]{Get
Started} quickstart. It walks the same five steps — load data, set parameters,
train, predict, and persist the model — so a reader coming from another XGBoost
binding can map each call directly.

@margin-note{Upstream's @emph{Python} quickstart now leads with the
scikit-learn @tt{XGBClassifier} estimator. This binding has no scikit-style
estimator; the @racket[train] / @racket[predict] pair shown here is the direct
analogue of the booster-level @tt{xgb.train} flow that the same upstream page
uses for R, Julia, and Scala.}

@section{From a data file}

XGBoost reads several on-disk formats. The classic one is
@hyperlink["https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html"]{LIBSVM
text}, where each line is a label followed by @tt{index:value} feature pairs.
@racket[make-dmatrix-from-uri] loads such a file into a @tech{DMatrix}; pass
@racket[#:format "libsvm"] so XGBoost parses it as LIBSVM rather than guessing
from the extension:

@racketblock[
(require xgboost)

(code:comment "Python: dtrain = xgb.DMatrix('train.svm.txt')")
(define dtrain (make-dmatrix-from-uri "train.svm.txt" #:format "libsvm"))
(define dtest  (make-dmatrix-from-uri "test.svm.txt"  #:format "libsvm"))
]

With the data loaded, set a small parameter dictionary and train for a fixed
number of boosting rounds. @racket[#:params] takes the same key/value pairs as
the Python @tt{param} dict, and @racket[#:rounds] is @tt{num_round}:

@racketblock[
(code:comment "Python: param = {'max_depth': 2, 'eta': 1,")
(code:comment "                 'objective': 'binary:logistic'}")
(code:comment "        bst = xgb.train(param, dtrain, num_round=2)")
(define bst
  (train dtrain
         #:params '((max_depth . 2)
                    (eta       . 1.0)
                    (objective . "binary:logistic"))
         #:rounds 2))
]

Prediction runs the trained booster over a DMatrix. For
@racket["binary:logistic"] the results are class-1 probabilities:

@racketblock[
(code:comment "Python: preds = bst.predict(dtest)")
(define preds (predict bst dtest))
]

Finally, persist the model and read it back. XGBoost picks the format from the
file extension (use @filepath{.json} or @filepath{.ubj}); a reloaded booster
predicts identically to the original:

@racketblock[
(code:comment "Python: bst.save_model('model.json')")
(save-model bst "model.json")

(code:comment "Python: bst = xgb.Booster(); bst.load_model('model.json')")
(define reloaded (load-model "model.json"))
]

A complete, runnable version of this flow — staging the LIBSVM files in a
temporary directory and asserting the round-trip — is
@filepath{examples/27-get-started.rkt} in the package source.

@section{From in-memory Racket data}

You rarely have a LIBSVM file on hand when experimenting. @racket[make-dmatrix]
builds a DMatrix straight from Racket values — a list of row lists, a vector of
row vectors, or a flat row-major sequence — with labels attached via
@racket[#:labels]:

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
entries in @racket[#:params]; @secref["parameters"] covers both styles. The
@racket[train] / @racket[predict] pair is the single training interface, so
this in-memory path and the file-based path above differ only in how the
DMatrix is built.

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
 @item{@secref["objectives"] — supplying a custom Racket objective.}
 @item{@secref["inspection"] — tree dumps and feature importance.}
 @item{@secref["config"] — global configuration, snapshots, and GPU training.}
]

For the precise contract of every procedure, see the @secref["reference"].
