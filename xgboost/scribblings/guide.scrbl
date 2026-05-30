#lang scribble/manual

@(require (for-label racket/base
                     "../main.rkt"))

@title{XGBoost: A User Guide}

@author{The xgboost Racket package}

This guide is a task-oriented tour of the high-level @racketmodname[xgboost] API.
It follows the same arc as the
@hyperlink["https://xgboost.readthedocs.io/en/stable/python/python_intro.html"]{Python
package introduction} — loading data, setting parameters, training, predicting,
and inspecting a model — adapted to Racket. For the precise contract of every
procedure, see the @racketmodname[xgboost] API reference.

@section{Installation}

Install from the Racket package catalog:

@commandline{raco pkg install xgboost}

The package ships a prebuilt native library and selects the right one for your
platform at install time; on Linux it prefers a CUDA-enabled build when one is
available and falls back to the CPU build otherwise. No XGBoost installation of
your own is required.

Everything in this guide uses the default high-level module:

@racketblock[(require xgboost)]

@section{A complete example}

A full train-and-predict run is short. Build a @tech{DMatrix} with labels, fit a
booster, and predict:

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

The @racket[train]/@racket[predict] pair is the single training interface; there
is no separate scikit-learn-style estimator. The sections below break the
workflow down step by step.

@include-section["guide/data.scrbl"]
@include-section["guide/parameters.scrbl"]
@include-section["guide/training.scrbl"]
@include-section["guide/prediction.scrbl"]
@include-section["guide/objectives.scrbl"]
@include-section["guide/inspection.scrbl"]
@include-section["guide/config.scrbl"]
