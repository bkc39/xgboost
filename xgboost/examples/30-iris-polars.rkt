#lang scribble/lp2

@(require (for-label racket/base
                     racket/list
                     racket/sequence
                     xgboost
                     (only-in polars dataframe?)))

@section[#:tag "ex-iris-polars"]{Iris with Polars}

This mirrors XGBoost's
@hyperlink["https://xgboost.readthedocs.io/en/release_3.2.0/get_started.html"]{Python
Getting Started}, but reads the data with @racketmodname[polars] and trains
straight off the @tech{DataFrame}: @racket[make-dmatrix], @racket[train], and
@racket[predict] all accept a Polars @racket[dataframe?] anywhere a
@tech{DMatrix} is expected (see @racket[dataframe->dmatrix]).

The upstream snippet passes @racket["binary:logistic"] to an
@tt{XGBClassifier} on the three-class iris problem --- a quirk that happens to
work because scikit-learn relabels internally. We do the honest thing and use
@racket["multi:softprob"] with @racket[#:num-class] @racket[3], which yields a
per-row probability vector we @racket[argmax] into a class.

@chunk[<r30-require>
(require xgboost
         ;; `filter` clashes with racket/base, so we bring polars' in renamed.
         (only-in polars read-csv write-parquet read-parquet
                  ref len select series (filter df-filter))
         racket/file
         racket/list
         racket/sequence
         racket/set
         racket/runtime-path)]

@chunk[<r30-provide>
(provide run-example)]

@bold{Setup.} The iris CSV ships next to this example; the four measurements are
the features and @tt{species} is the (string) target.

@chunk[<r30-setup>
  (define-runtime-path iris-csv "data/iris.csv")
  (define features
    '("sepal_length" "sepal_width" "petal_length" "petal_width"))
  (define species->class
    (hash "Iris-setosa" 0 "Iris-versicolor" 1 "Iris-virginica" 2))]

@bold{Helpers.} A couple of small utilities: reading a column into a Racket list,
a reproducible shuffled train/test split expressed as boolean masks, and
@racket["multi:softprob"] post-processing (flat probabilities to per-row
@racket[argmax] to accuracy).

@chunk[<r30-helpers>
  (define (column->list df name)
    (define c (ref df name))
    (for/list ([i (in-range (len c))]) (ref c i)))

  ;; A seeded shuffle, returned as (values train-mask test-mask): two boolean
  ;; lists we feed to polars `filter` (and to the label lists) so the row
  ;; selection stays identical across the frame and the labels.
  (define (split-masks n test-frac seed)
    (define rng (vector->pseudo-random-generator (vector seed 1 2 3 4 5)))
    (define idx
      (parameterize ([current-pseudo-random-generator rng])
        (shuffle (range n))))
    (define n-test (inexact->exact (round (* n test-frac))))
    (define test-set (list->seteqv (take idx n-test)))
    (values (for/list ([i (in-range n)]) (not (set-member? test-set i)))
            (for/list ([i (in-range n)]) (set-member? test-set i))))

  (define (mask-filter xs mask)
    (for/list ([x (in-list xs)] [m (in-list mask)] #:when m) x))

  (define (argmax-class probs)
    (for/fold ([best 0] [bv (car probs)] #:result best)
              ([p (in-list probs)] [i (in-naturals)])
      (if (> p bv) (values i p) (values best bv))))

  ;; `predict` on a softprob booster returns one flat list of length 3*N; slice
  ;; it back into per-row probability triples and take the argmax class.
  (define (predict-classes bst df)
    (for/list ([row (in-slice 3 (predict bst df))]) (argmax-class row)))

  (define (accuracy preds labels)
    (/ (for/sum ([p (in-list preds)] [y (in-list labels)]) (if (= p y) 1 0))
       (length labels)))]

@bold{Load and split.} @racket[read-csv] returns a @racket[dataframe?];
@racket[select] keeps just the numeric feature columns (XGBoost takes numbers,
not the @tt{species} strings), and the boolean masks partition both the frame
(via @racket[filter]) and the integer labels.

@chunk[<r30-data>
  (define df (read-csv iris-csv))
  (define labels
    (map (lambda (s) (hash-ref species->class s)) (column->list df "species")))
  (define feature-df (select df features))
  (define-values (train-mask test-mask) (split-masks (len df) 0.2 42))
  (define train-df (df-filter feature-df (series train-mask)))
  (define test-df  (df-filter feature-df (series test-mask)))
  (define y-train (mask-filter labels train-mask))
  (define y-test  (mask-filter labels test-mask))]

@bold{Fit and predict.} @racket[train] takes the DataFrame directly; @racket[#:labels]
supplies the targets (a column name or, as here, a plain sequence).

@chunk[<r30-fit>
  (define bst
    (train train-df
           #:labels y-train
           #:objective "multi:softprob"
           #:num-class 3
           #:max-depth 3
           #:eta 0.3
           #:verbosity 0
           #:rounds 20))
  (define preds (predict-classes bst test-df))]

@bold{Parquet round-trip.} Writing the test frame to Parquet and reading it back
yields identical predictions --- a quick check that the columnar path is
lossless.

@chunk[<r30-parquet>
  (define parquet-path (make-temporary-file "iris-~a.parquet"))
  (write-parquet test-df parquet-path)
  (define preds-parquet (predict-classes bst (read-parquet parquet-path)))
  (delete-file parquet-path)]

@racket[run-example] returns the train/test sizes, the test accuracy, and whether
the Parquet round-trip matched. The harness @filepath{test/30-iris-polars.rkt}
prints a one-line summary and asserts iris is classified comfortably.

@chunk[<r30-run>
(define (run-example)
  <r30-data>
  <r30-fit>
  <r30-parquet>
  (values (length y-train)
          (length y-test)
          (accuracy preds y-test)
          (equal? preds preds-parquet)))]

@chunk[<*>
  <r30-require>
  <r30-provide>
  <r30-setup>
  <r30-helpers>
  <r30-run>]
