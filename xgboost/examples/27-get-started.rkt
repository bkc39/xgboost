#lang scribble/lp2

@(require (for-label racket/base
                     xgboost
                     xgboost/private/demo-utils))

@section[#:tag "ex-get-started"]{Get Started}

This is the Racket counterpart of XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/stable/get_started.html"]{Python
quickstart}. The upstream snippet is:

@verbatim{
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], test_size=.2)
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1)
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
}

The binding has no scikit-style @tt{XGBClassifier} estimator, so instead of
@tt{.fit}/@tt{.predict} we build a @tech{DMatrix} and call @racket[train] /
@racket[predict] --- the same flow the upstream R, Julia, and Scala quickstarts
use. @racket[load-iris] and @racket[train-test-split] come from
@racketmodname[xgboost/private/demo-utils] (@racket[load-iris] downloads the UCI
dataset, falling back to a bundled copy offline). Iris has three classes, so we
use @racket["multi:softmax"] with @racket[#:num-class] @racket[3] --- then
@racket[predict] returns class indices, like sklearn's @tt{.predict} returns
labels.

@chunk[<r27-require>
(require xgboost
         xgboost/private/demo-utils)]

@chunk[<r27-provide>
(provide run-example)]

@bold{Accuracy.} A small helper, since unlike sklearn we score by hand:

@chunk[<r27-accuracy>
  (define (accuracy preds labels)
    (/ (for/sum ([p (in-list preds)] [y (in-list labels)])
         (if (= (inexact->exact (round p)) y) 1 0))
       (length labels)))]

@bold{Load and split.} @racket[load-iris] returns the @tt{150 × 4} feature rows
and integer labels; @racket[train-test-split] holds out 20% with a fixed seed
for reproducibility:

@chunk[<r27-data>
  (define-values (X y) (load-iris))
  (define-values (X-train X-test y-train y-test)
    (train-test-split X y #:test-size 0.2 #:seed 42))]

@bold{Fit and predict.} @racket[make-dmatrix] accepts the row lists directly;
two shallow rounds with @racket[#:eta] @racket[1.0] are plenty for iris:

@chunk[<r27-fit>
  (define bst
    (train (make-dmatrix X-train #:labels y-train)
           #:num-class 3
           #:objective "multi:softmax"
           #:max-depth 2
           #:eta 1.0
           #:verbosity 0
           #:rounds 2))
  (define preds (predict bst (make-dmatrix X-test)))]

@racket[run-example] returns the training size, the predictions, and the test
accuracy. The harness @filepath{test/27-get-started.rkt} prints a one-line
summary and asserts iris is comfortably classified:

@racketblock[
(code:comment "get-started: 120 train / 30 test, test accuracy 0.967")
]

@chunk[<r27-run>
(define (run-example)
  <r27-data>
  <r27-fit>
  (values (length X-train) preds (accuracy preds y-test)))]

@chunk[<*>
  <r27-require>
  <r27-provide>
  <r27-accuracy>
  <r27-run>]
