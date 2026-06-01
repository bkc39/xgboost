#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     "../../main.rkt"
                     "../../private/demo-utils.rkt"))

@title[#:tag "getting-started"]{Getting Started}

This chapter trains a classifier on the classic
@hyperlink["https://archive.ics.uci.edu/ml/datasets/iris"]{iris} dataset — load
the data, split it, fit a booster, and predict. @racket[load-iris] and
@racket[train-test-split] are demo helpers from
@racketmodname[xgboost/private/demo-utils]; everything else is the high-level
@racketmodname[xgboost] API.

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

A complete, runnable, assertion-backed version is
@filepath{examples/27-get-started.rkt} in the package source.
