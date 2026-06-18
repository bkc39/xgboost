#lang scribble/manual

@(require (for-label ffi/vector
                     racket/base
                     racket/contract
                     (only-in polars dataframe?)
                     "../../main.rkt"))

@title[#:tag "data"]{Data Interface}

XGBoost trains and predicts over a @deftech{DMatrix}: a dense or sparse feature
matrix together with optional per-row metadata such as labels and weights. In
the high-level API a DMatrix is an opaque Racket value whose native handle is
reclaimed by the garbage collector; you never free it by hand.

@section{Building a DMatrix from Racket data}

The most direct constructor is @racket[make-dmatrix]. It accepts a list of row
lists, a vector of row vectors, or a flat row-major sequence when you also pass
@racket[#:nrow] and @racket[#:ncol]:

@racketblock[
(require xgboost)

(code:comment "nested rows — shape is inferred")
(make-dmatrix '((1.0 2.0 0.5)
                (2.0 1.0 1.5)
                (3.0 0.5 0.0)))

(code:comment "flat row-major data — shape supplied explicitly")
(require ffi/vector)
(make-dmatrix (f32vector 1.0 2.0 3.0
                         4.0 5.0 6.0)
              #:nrow 2 #:ncol 3)
]

You can attach labels (and weights) at construction time:

@racketblock[
(make-dmatrix '((1.0 2.0 0.5)
                (2.0 1.0 1.5)
                (3.0 0.5 0.0)
                (0.5 3.0 2.0))
              #:labels '(3.5 3.5 6.5 2.0))
]

@section{Polars DataFrames}

A Polars @deftech{DataFrame} (a @racket[dataframe?] from
@racketmodname[polars]) can be used wherever a DMatrix is expected:
@racket[make-dmatrix], @racket[train], and @racket[predict] all detect a
DataFrame and route it through @racket[dataframe->dmatrix]. Every column becomes a
feature unless you name a label (or weight) column:

@racketblock[
(require xgboost
         (only-in polars read-csv select))

(code:comment "label by column name — \"species\" is pulled out of the features")
(define df (read-csv "iris.csv"))
(make-dmatrix df #:labels "species")

(code:comment "or supply labels as a sequence alongside a feature-only frame")
(define features
  '("sepal_length" "sepal_width" "petal_length" "petal_width"))
(make-dmatrix (select df features) #:labels '(0 1 2 0))
]

Columns must be numeric: a string column (such as a raw class label) raises
unless excluded via @racket[#:labels] or a feature selection. For the full bridge
contract, including @racket[#:feature-columns] and @racket[#:weights], see
@racket[dataframe->dmatrix]; for an end-to-end model see the
@secref["ex-iris-polars"] example.

@section{Missing values}

XGBoost represents a missing entry with a sentinel value. By default that
sentinel is @racket[+nan.0]; pass @racket[#:missing] to choose a different
marker. Here @racket[-1.0] marks the holes, so the materialized matrix shows
@racket[+nan.0] where the data had @racket[-1.0] (or, for sparse inputs, where
no entry was supplied):

@racketblock[
(make-dmatrix (f32vector 1.0 2.0 3.0
                         4.0 5.0 6.0)
              #:nrow 2 #:ncol 3
              #:missing -1.0)
]

@section{Sparse and columnar inputs}

For data that is already in a native layout, build directly from CSR, CSC, or
column-major storage. All three matrices below describe the same 2×3 data with
two missing cells:

@racketblock[
(code:comment "CSR: row offsets, column indices, values, ncol")
(make-dmatrix-from-csr (u64vector 0 2 4)
                       (u32vector 0 2 1 2)
                       (f32vector 1.0 3.0 5.0 6.0)
                       3
                       -1.0)

(code:comment "CSC: column offsets, row indices, values, nrow")
(make-dmatrix-from-csc (u64vector 0 1 2 4)
                       (u32vector 0 1 0 1)
                       (f32vector 1.0 5.0 3.0 6.0)
                       2
                       -1.0)

(code:comment "Columnar: one f32vector per column")
(make-dmatrix-from-columnar (list (f32vector 1.0 4.0)
                                  (f32vector 2.0 5.0)
                                  (f32vector 3.0 6.0))
                            -1.0)
]

@section{Loading from a file}

@racket[make-dmatrix-from-uri] reads an XGBoost-supported file or URI. Pass
@racket[#:format] to disambiguate when the extension is not enough:

@racketblock[
(make-dmatrix-from-uri "train.libsvm" #:format "libsvm")
]

A DMatrix can also be written to XGBoost's binary buffer format with
@racket[dmatrix-save-binary!] and reloaded later with
@racket[make-dmatrix-from-uri].

@section{Inspecting a DMatrix}

@racket[dmatrix-rows] and @racket[dmatrix-cols] report the shape,
@racket[dmatrix->list] materializes the contents as nested lists (missing cells
appear as @racket[+nan.0]), and @racket[dmatrix-show] prints a human-readable
rendering.

@section{Labels, weights, and feature metadata}

Metadata can be set after construction. Setters accept lists, vectors, or typed
vectors and coerce internally; getters read the fields back as Racket data:

@racketblock[
(define dm
  (make-dmatrix (f32vector 1.0 2.0
                           3.0 4.0)
                #:nrow 2 #:ncol 2))

(dmatrix-set-label! dm '(0.25 0.75))
(dmatrix-set-feature-names! dm '("height" "weight"))
(dmatrix-set-feature-types! dm '("q" "q"))

(dmatrix-feature-names dm)  (code:comment "=> '(\"height\" \"weight\")")
(dmatrix-label dm)          (code:comment "=> '(0.25 0.75)")
]

Other well-known fields have dedicated setters: @racket[dmatrix-set-weight!],
@racket[dmatrix-set-base-margin!], @racket[dmatrix-set-group!] (for ranking),
and @racket[dmatrix-set-label-lower-bound!] /
@racket[dmatrix-set-label-upper-bound!] (for AFT survival objectives). Each
identifier above links to its full contract in the @secref["reference"].
