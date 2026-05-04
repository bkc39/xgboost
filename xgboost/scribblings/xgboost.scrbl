#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     racket/contract
                     "../main.rkt"))

@title{XGBoost}

@defmodule[xgboost]

This module provides the high-level Racket API for XGBoost. It accepts ordinary
Racket data for common training and prediction workflows, while keeping native
XGBoost handles behind opaque Racket values.

For lower-level access, use @racketmodname[xgboost/ffi]. For direct C FFI
bindings, use @racketmodname[xgboost/ffi/raw].

The @racketmodname[xgboost/ffi] module also exposes lower-level DMatrix
constructors for URI loading, dense array-interface input, CSR, CSC, and
columnar array-interface input. These are intended for callers that already
work with native-style buffers or XGBoost JSON array-interface strings.
It also exposes local DMatrix metadata and dataset operations such as feature
names/types, uint info, row slicing, binary DMatrix saving, and quantile-cut
inspection.
Booster inspection APIs cover lifecycle queries, JSON config round-trips,
attributes, feature info, model dumps, and feature scores.
CPU inplace prediction APIs support dense, CSR, and columnar array-interface
inputs for serving-style prediction without constructing a DMatrix first.

@section{Example}

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

@section{Data}

@defproc[(dmatrix? [v any/c]) boolean?]{
Returns @racket[#t] when @racket[v] is a high-level DMatrix value produced by
@racket[make-dmatrix].
}

@defproc[(make-dmatrix [data any/c]
                       [#:nrow nrow (or/c #f exact-positive-integer?) #f]
                       [#:ncol ncol (or/c #f exact-positive-integer?) #f]
                       [#:missing missing real? +nan.0]
                       [#:labels labels (or/c #f any/c) #f]
                       [#:weights weights (or/c #f any/c) #f])
         dmatrix?]{
Creates an XGBoost DMatrix.

@racket[data] may be one of:

@itemlist[
 @item{a list of row lists}
 @item{a vector of row vectors}
 @item{a flat row-major list, vector, or @racket[f32vector?], when both
       @racket[#:nrow] and @racket[#:ncol] are supplied}
]

Rows must be rectangular. Labels and weights may be lists, vectors, or
@racket[f32vector?] values, and their lengths must match the inferred row
count.
}

@section{Training}

@defproc[(booster? [v any/c]) boolean?]{
Returns @racket[#t] when @racket[v] is a high-level Booster value produced by
@racket[train], @racket[load-model], or @racket[load-model-from-bytes].
}

@defproc[(train [dtrain dmatrix?]
                [#:params params any/c '()]
                [#:rounds rounds exact-nonnegative-integer? 10]
                [#:evals evals (listof (cons/c string? dmatrix?)) '()]
                [#:objective objective (or/c #f any/c) #f]
                [#:eta eta (or/c #f any/c) #f]
                [#:max-depth max-depth (or/c #f any/c) #f]
                [#:num-class num-class (or/c #f any/c) #f]
                [#:eval-metric eval-metric (or/c #f any/c) #f]
                [#:verbosity verbosity (or/c #f any/c) #f])
         booster?]{
Trains a Booster for @racket[rounds] boosting rounds.

@racket[params] may be a hash or association list. Parameter keys may be
strings, symbols, or keywords. Symbol and keyword keys are converted to
XGBoost-style names by replacing hyphens with underscores. Parameter values
are converted to strings before reaching the FFI layer.

Keyword conveniences such as @racket[#:objective] and @racket[#:max-depth] are
merged after @racket[params], so they override entries with the same XGBoost
parameter name.

The returned Booster retains references to @racket[dtrain] and all evaluation
DMatrices so native cache inputs remain live while the Booster is live.
}

@defproc[(predict [booster booster?]
                  [dmat dmatrix?]
                  [#:output output (or/c 'value 'margin 'contribs
                                          'approx-contribs 'interactions
                                          'approx-interactions 'leaf)
                            'value]
                  [#:iteration-end iteration-end exact-nonnegative-integer? 0]
                  [#:as as (or/c 'list 'f32vector) 'list])
         (or/c (listof real?) f32vector?)]{
Runs prediction for @racket[dmat].

By default, predictions are returned as a list. Pass @racket[#:as 'f32vector]
to receive the Racket @racket[f32vector?] copied from XGBoost output.

@racket[#:iteration-end] limits prediction to the first N boosting rounds;
@racket[0] means all available rounds.
}

@section{Evaluation}

@defproc[(eval-one-iter [booster booster?]
                        [iter exact-integer?]
                        [evals (listof (cons/c string? dmatrix?))])
         string?]{
Evaluates @racket[booster] at @racket[iter] for the named DMatrices in
@racket[evals], returning XGBoost's metric line.
}

@defproc[(parse-eval-line [line string?]) (hash/c string? real?)]{
Parses an XGBoost metric line into a hash from @racket["name-metric"] strings
to numeric values.
}

@section{Model IO}

@defproc[(save-model [booster booster?] [path path-string?]) void?]{
Saves @racket[booster] to @racket[path]. XGBoost chooses the model format from
the file extension.
}

@defproc[(load-model [path path-string?]) booster?]{
Loads a Booster from @racket[path].
}

@defproc[(save-model-to-bytes [booster booster?]
                              [#:format format (or/c "json" "ubj") "ubj"])
         bytes?]{
Serializes @racket[booster] to bytes in JSON or UBJSON format.
}

@defproc[(load-model-from-bytes [data bytes?]) booster?]{
Loads a Booster from bytes produced by @racket[save-model-to-bytes].
}

@section{Version}

@defproc[(xgboost-version) string?]{
Returns the linked XGBoost version string.
}

@defproc[(xgboost-build-info) string?]{
Returns XGBoost build information as a JSON string.
}

@section{Process Configuration}

@defproc[(xgboost-get-global-config) string?]{
Returns XGBoost's process-global configuration as a JSON string.
}

@defproc[(xgboost-set-global-config! [config string?]) void?]{
Sets XGBoost's process-global configuration from a JSON string.
}

@defproc[(xgboost-register-log-callback! [callback (-> string? any/c)])
         void?]{
Registers a process-global XGBoost log callback.

The callback receives log messages as strings. Since the registration is
process-global, callers should treat it as shared mutable process state.
}
