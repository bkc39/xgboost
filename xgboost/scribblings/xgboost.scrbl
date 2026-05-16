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

For lower-level access, use @racketmodname[xgboost/foreign]. For direct C FFI
bindings, use @racketmodname[xgboost/foreign/raw].

The @racketmodname[xgboost/foreign] module also exposes lower-level DMatrix
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
Custom objective training is available by supplying gradient and Hessian
vectors for one boosting iteration at a time.

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

DMatrix lifetimes are managed by Racket's GC: the underlying XGBoost handle
is reclaimed once the wrapper is unreachable. There is no public free
operation; if you need deterministic release for a long-lived workload,
import @racket[(submod xgboost/foreign unsafe)].
}

@defproc[(make-dmatrix-from-csr [indptr u64vector?]
                                [indices u32vector?]
                                [data f32vector?]
                                [ncol exact-positive-integer?]
                                [missing real? +nan.0])
         dmatrix?]{
Creates a DMatrix from CSR storage: @racket[indptr] holds row offsets,
@racket[indices] holds column indices, and @racket[data] holds the
corresponding values. Missing entries materialize as @racket[missing].
}

@defproc[(make-dmatrix-from-csc [indptr u64vector?]
                                [indices u32vector?]
                                [data f32vector?]
                                [nrow exact-positive-integer?]
                                [missing real? +nan.0])
         dmatrix?]{
Creates a DMatrix from CSC storage: @racket[indptr] holds column offsets,
@racket[indices] holds row indices, and @racket[data] holds values.
}

@defproc[(make-dmatrix-from-columnar [columns (listof f32vector?)]
                                     [missing real? +nan.0])
         dmatrix?]{
Creates a DMatrix from column-major @racket[f32vector?] columns. All columns
must have the same length.
}

@defproc[(make-dmatrix-from-uri [uri-or-path (or/c path-string? string?)]
                                [#:format format (or/c #f "libsvm" "csv") #f]
                                [#:silent? silent? any/c #t])
         dmatrix?]{
Loads a DMatrix from a file path or XGBoost URI. When @racket[#:format] is a
string, it is appended as a query parameter (for example, libsvm-format
files load as @racket[#:format "libsvm"]).
}

@defproc[(dmatrix-rows [dm dmatrix?]) exact-nonnegative-integer?]{
Returns the row count of @racket[dm].
}

@defproc[(dmatrix-cols [dm dmatrix?]) exact-nonnegative-integer?]{
Returns the column count of @racket[dm].
}

@defproc[(dmatrix->list [dm dmatrix?]) (listof (listof real?))]{
Materializes @racket[dm] as a list of row lists. Missing entries appear as
@racket[+nan.0].
}

@defproc[(dmatrix-show [dm dmatrix?] [port output-port? (current-output-port)])
         void?]{
Writes a human-readable rendering of @racket[dm] to @racket[port].
}

@defproc[(dmatrix-slice [dm dmatrix?]
                        [indices (or/c list? vector? s32vector?)]
                        [#:allow-groups? allow-groups? any/c #f])
         dmatrix?]{
Returns a fresh DMatrix containing the rows of @racket[dm] selected by
@racket[indices]. Pass @racket[#:allow-groups? #t] to slice a DMatrix that
carries ranking group information.
}

@defproc[(dmatrix-save-binary! [dm dmatrix?]
                               [path path-string?]
                               [#:silent? silent? any/c #t])
         void?]{
Writes @racket[dm] to @racket[path] in XGBoost's binary buffer format.
Reload with @racket[make-dmatrix-from-uri].
}

@section{DMatrix Metadata}

These helpers attach and retrieve the well-known XGBoost label/weight
fields, ranking groups, and feature info. Label/weight/group setters accept
lists, vectors, or typed vectors and coerce internally.

@defproc[(dmatrix-set-label! [dm dmatrix?] [xs any/c]) void?]{
Sets the @racket["label"] float info field.
}

@defproc[(dmatrix-set-weight! [dm dmatrix?] [xs any/c]) void?]{
Sets the @racket["weight"] float info field.
}

@defproc[(dmatrix-set-base-margin! [dm dmatrix?] [xs any/c]) void?]{
Sets the @racket["base_margin"] float info field.
}

@defproc[(dmatrix-set-label-lower-bound! [dm dmatrix?] [xs any/c]) void?]{
Sets the @racket["label_lower_bound"] field used by AFT survival objectives.
}

@defproc[(dmatrix-set-label-upper-bound! [dm dmatrix?] [xs any/c]) void?]{
Sets the @racket["label_upper_bound"] field used by AFT survival objectives.
Use @racket[+inf.0] for right-censored observations.
}

@defproc[(dmatrix-set-group! [dm dmatrix?] [sizes any/c]) void?]{
Sets the ranking group sizes. The cumulative @racket["group_ptr"] is
maintained by XGBoost and read back via @racket[dmatrix-group-ptr].
}

@defproc[(dmatrix-set-feature-names! [dm dmatrix?] [names (listof string?)]) void?]
@defproc[(dmatrix-set-feature-types! [dm dmatrix?] [types (listof string?)]) void?]{
Set the feature-name and feature-type metadata.
}

@deftogether[(@defproc[(dmatrix-label [dm dmatrix?]) (listof real?)]
              @defproc[(dmatrix-weight [dm dmatrix?]) (listof real?)]
              @defproc[(dmatrix-base-margin [dm dmatrix?]) (listof real?)]
              @defproc[(dmatrix-group-ptr [dm dmatrix?])
                       (listof exact-nonnegative-integer?)]
              @defproc[(dmatrix-feature-names [dm dmatrix?])
                       (listof string?)]
              @defproc[(dmatrix-feature-types [dm dmatrix?])
                       (listof string?)])]{
Read the corresponding metadata fields back as Racket data.
}

@defproc[(dmatrix-quantile-cut [dm dmatrix?])
         (values (listof exact-nonnegative-integer?) f32vector?)]{
Returns @racket[(values indptr data)] for the quantile cuts XGBoost computed
during hist-mode training. @racket[indptr] holds per-feature offsets into
@racket[data], whose final entry equals the total cut count.

Quantile cuts only exist after at least one round of @racket["tree_method"]
@racket["hist"] training; calling this before training returns empty
results.
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
                [#:objective-fn objective-fn
                                (or/c #f (-> f32vector? dmatrix? any)) #f]
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

When @racket[#:objective-fn] is supplied, each round runs a Racket-side
custom objective: @racket[train] computes margin predictions, calls
@racket[(objective-fn preds dtrain)] which must return
@racket[(values grad hess)] (lists, vectors, or @racket[f32vector?] values),
and feeds those gradients into one boosting iteration. Use @racket[#:rounds 0]
to construct an untrained Booster bound to @racket[dtrain] (and any
@racket[#:evals] entries) for manual stepping with
@racket[booster-update-one-iter!] or @racket[booster-train-one-iter!].

The returned Booster retains references to @racket[dtrain] and all evaluation
DMatrices so native cache inputs remain live while the Booster is live.
}

@defproc[(make-booster) booster?]{
Creates an empty Booster with no DMatrix cache. Useful for setting attributes
ahead of training, loading a JSON config, or preparing a target for
@racket[bytes->booster].
}

@defproc[(booster-cache [booster booster?]) (listof dmatrix?)]{
Returns the list of DMatrices the Booster retains alive (the training
DMatrix followed by any @racket[#:evals] DMatrices).
}

@defproc[(booster-num-feature [booster booster?]) exact-nonnegative-integer?]{
Returns the number of features the Booster was trained on.
}

@defproc[(booster-boosted-rounds [booster booster?])
         exact-nonnegative-integer?]{
Returns how many boosting rounds the Booster has been trained for.
}

@defproc[(booster-reset! [booster booster?]) void?]{
Releases internal training caches without resetting the trained model.
@racket[booster-boosted-rounds] is unchanged after reset.
}

@defproc[(booster-slice [booster booster?]
                        [begin-layer exact-integer?]
                        [end-layer exact-integer?]
                        [step exact-positive-integer? 1])
         booster?]{
Returns a fresh Booster containing only the trees in the half-open layer
range from @racket[begin-layer] (inclusive) to @racket[end-layer]
(exclusive). The result is independent of @racket[booster] and shares no
XGBoost-internal state.
}

@defproc[(booster-set-param! [booster booster?] [key any/c] [value any/c])
         void?]{
Sets a single XGBoost parameter. Keys and values are coerced through the
same rules as @racket[train]'s @racket[#:params].
}

@defproc[(booster-update-one-iter! [booster booster?]
                                   [iter exact-integer?]
                                   [dtrain dmatrix?])
         void?]{
Runs one XGBoost-built-in objective boosting round on @racket[dtrain].
}

@defproc[(booster-train-one-iter! [booster booster?]
                                  [iter exact-integer?]
                                  [dtrain dmatrix?]
                                  [grad any/c]
                                  [hess any/c])
         void?]{
Runs one custom-objective boosting round using user-supplied gradient and
Hessian vectors. Both vectors must have one entry per row of
@racket[dtrain] (or per row times num-class for multiclass objectives).
}

@section{Booster Attributes and Configuration}

@deftogether[(@defproc[(booster-set-attr! [booster booster?]
                                          [key string?] [value string?]) void?]
              @defproc[(booster-attr [booster booster?] [key string?])
                       (or/c #f string?)]
              @defproc[(booster-attr-names [booster booster?])
                       (listof string?)]
              @defproc[(booster-delete-attr! [booster booster?] [key string?])
                       void?])]{
Read, write, list, and delete user-defined string attributes on
@racket[booster]. @racket[booster-attr] returns @racket[#f] when the key is
absent.
}

@deftogether[(@defproc[(booster-set-feature-names! [booster booster?]
                                                   [names (listof string?)])
                       void?]
              @defproc[(booster-set-feature-types! [booster booster?]
                                                   [types (listof string?)])
                       void?]
              @defproc[(booster-feature-names [booster booster?])
                       (listof string?)]
              @defproc[(booster-feature-types [booster booster?])
                       (listof string?)])]{
Read and write the feature-name and feature-type metadata stored on
@racket[booster].
}

@defproc[(booster-config [booster booster?]) string?]{
Returns @racket[booster]'s configuration as a JSON string. The shape is
XGBoost's wire format; treat it as opaque.
}

@defproc[(booster-set-config! [booster booster?] [config string?]) void?]{
Loads a JSON configuration produced by @racket[booster-config].
}

@defproc[(booster-dump [booster booster?]
                       [#:format format (or/c "text" "json" "dot") "text"]
                       [#:with-stats? with-stats? any/c #f]
                       [#:feature-names feature-names (or/c #f (listof string?)) #f]
                       [#:feature-types feature-types (or/c #f (listof string?)) #f])
         (listof string?)]{
Returns one string per tree in @racket[booster]. Pass
@racket[#:feature-names] and @racket[#:feature-types] together to substitute
human-readable names into the dump.
}

@defproc[(booster-feature-score [booster booster?]
                                [#:importance-type importance-type string?
                                                   "weight"]
                                [#:feature-names feature-names
                                                 (or/c #f (listof string?))
                                                 #f]
                                [#:config config (or/c #f string?) #f])
         (hash/c symbol? any/c)]{
Returns a hash with keys @racket['features], @racket['shape], and
@racket['scores] describing per-feature importance. Pass
@racket[#:importance-type] to choose @racket["weight"], @racket["gain"], or
similar XGBoost importance modes.
}

@section{Inplace Prediction}

These variants run prediction directly against ordinary Racket data without
constructing a DMatrix. They share the same @racket[#:output],
@racket[#:iteration-end], and @racket[#:as] keywords as @racket[predict].

@defproc[(predict-from-dense [booster booster?]
                             [data any/c]
                             [#:nrow nrow (or/c #f exact-positive-integer?) #f]
                             [#:ncol ncol (or/c #f exact-positive-integer?) #f]
                             [#:missing missing real? +nan.0]
                             [#:output output (or/c 'value 'margin 'contribs
                                                    'approx-contribs
                                                    'interactions
                                                    'approx-interactions
                                                    'leaf) 'value]
                             [#:iteration-end iteration-end
                                              exact-nonnegative-integer? 0]
                             [#:as as (or/c 'list 'f32vector) 'list])
         (or/c (listof real?) f32vector?)]{
Predicts on a dense input shaped like @racket[make-dmatrix]'s @racket[data]
argument.
}

@defproc[(predict-from-csr [booster booster?]
                           [indptr u64vector?]
                           [indices u32vector?]
                           [data f32vector?]
                           [ncol exact-positive-integer?]
                           [#:missing missing real? +nan.0]
                           [#:output output (or/c 'value 'margin 'contribs
                                                  'approx-contribs
                                                  'interactions
                                                  'approx-interactions
                                                  'leaf) 'value]
                           [#:iteration-end iteration-end
                                            exact-nonnegative-integer? 0]
                           [#:as as (or/c 'list 'f32vector) 'list])
         (or/c (listof real?) f32vector?)]{
Predicts on CSR-encoded input.
}

@defproc[(predict-from-columnar [booster booster?]
                                [columns (listof f32vector?)]
                                [#:missing missing real? +nan.0]
                                [#:output output (or/c 'value 'margin 'contribs
                                                       'approx-contribs
                                                       'interactions
                                                       'approx-interactions
                                                       'leaf) 'value]
                                [#:iteration-end iteration-end
                                                 exact-nonnegative-integer? 0]
                                [#:as as (or/c 'list 'f32vector) 'list])
         (or/c (listof real?) f32vector?)]{
Predicts on column-major @racket[f32vector?] columns.
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

@section{Booster Snapshots}

@racket[save-model] / @racket[load-model] persist only the trained tree
ensemble. Snapshots additionally capture XGBoost's internal training caches,
so a restored Booster can resume per-iteration updates in lockstep with the
original.

@defproc[(booster->bytes [booster booster?]) bytes?]{
Serializes @racket[booster]'s full state, including training caches.
}

@defproc[(bytes->booster [data bytes?]) booster?]{
Reconstructs a Booster from bytes produced by @racket[booster->bytes]. The
restored Booster has an empty DMatrix cache; pass training data explicitly
to @racket[booster-update-one-iter!] when resuming.
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
