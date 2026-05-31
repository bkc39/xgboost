#lang scribble/manual

@(require (for-label racket/base
                     "../main.rkt"))

@title[#:tag "user-guide"]{User Guide}

This guide is a task-oriented tour of the high-level @racketmodname[xgboost] API.
It follows the same arc as the
@hyperlink["https://xgboost.readthedocs.io/en/stable/python/python_intro.html"]{Python
package introduction} — loading data, setting parameters, training, predicting,
and inspecting a model — adapted to Racket. For the precise contract of every
procedure, see the @secref["reference"].

@section{Installation}

Install from the Racket package catalog:

@commandline{raco pkg install xgboost}

The package ships a prebuilt native library and selects the right one for your
platform at install time; on Linux it prefers a CUDA-enabled build when one is
available and falls back to the CPU build otherwise. No XGBoost installation of
your own is required.

Everything in this guide uses the default high-level module:

@racketblock[(require xgboost)]

@include-section["guide/getting-started.scrbl"]
@include-section["guide/data.scrbl"]
@include-section["guide/parameters.scrbl"]
@include-section["guide/training.scrbl"]
@include-section["guide/prediction.scrbl"]
@include-section["guide/ranking.scrbl"]
@include-section["guide/objectives.scrbl"]
@include-section["guide/recipes.scrbl"]
@include-section["guide/inspection.scrbl"]
@include-section["guide/config.scrbl"]
