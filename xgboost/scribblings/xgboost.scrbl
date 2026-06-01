#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     "../main.rkt"))

@title{XGBoost}

@author[(author+email "bkc" "bkschemer@gmail.com")]

Racket bindings for
@hyperlink["https://xgboost.readthedocs.io/"]{XGBoost}, the gradient-boosting
library.

@racketmodname[xgboost] is the high-level API for ordinary Racket data: build a
DMatrix, @racket[train] a booster, @racket[predict], and persist the model. It
accepts lists, vectors, and @racket[f32vector]s for common training and
prediction workflows while keeping native XGBoost handles behind opaque Racket
values.

For lower-level access, use @racketmodfont{xgboost/foreign}, a contracted
wrapper that exposes additional DMatrix constructors, metadata and dataset
operations, booster inspection, CPU in-place prediction, and custom-objective
training. For direct C FFI bindings, use @racketmodfont{xgboost/foreign/raw}.

This documentation has two parts: a task-oriented @secref["user-guide"] that
mirrors the upstream XGBoost tutorials, and a complete @secref["reference"].

@include-section["xgboost-guide.scrbl"]
@include-section["reference.scrbl"]
