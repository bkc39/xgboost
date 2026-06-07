#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/string
                     xgboost))

@section[#:tag "ex-iris"]{Iris: a full classification pipeline}

This example walks the whole real-data path on the classic
@hyperlink["https://archive.ics.uci.edu/ml/datasets/iris"]{UCI Iris} dataset
(150 rows, 4 features, 3 species) with no network or filesystem dependency --- the
CSV is embedded as a string literal at the end of the file. The flow is: parse
the CSV, make a deterministic train/test split, build @tech{DMatrix}es, train
@racket["multi:softprob"] while logging @racket["mlogloss"], then predict on the
held-out set and report accuracy and a confusion matrix.

@chunk[<r06-require>
(require ffi/vector
         racket/list
         racket/string
         xgboost)]

@chunk[<r06-provide>
(provide run-example)]

@bold{Labels.} Map the species strings to class indices:

@chunk[<r06-constants>
  (define class-index
    (hash "Iris-setosa"     0
          "Iris-versicolor" 1
          "Iris-virginica"  2))]

@bold{Parsing.} Each non-empty line is four feature numbers and a species name;
@racket[parse-iris] returns a list of @tt{(features . label)} pairs:

@chunk[<r06-parse>
  (define (parse-iris csv)
    (for/list ([line (in-list (string-split csv "\n"))]
               #:when (positive? (string-length (string-trim line))))
      (define parts (string-split line ","))
      (cons (for/list ([p (in-list (take parts 4))]) (string->number p))
            (hash-ref class-index (list-ref parts 4)))))]

@bold{Splitting.} Every 5th row (in CSV order, which strides through the three
species) goes to test --- a deterministic, class-balanced 120/30 split:

@chunk[<r06-split>
  (define (split-rows rows)
    (for/fold ([train '()] [test '()] #:result (values (reverse train)
                                                        (reverse test)))
              ([row (in-list rows)] [i (in-naturals)])
      (if (zero? (modulo i 5))
          (values train (cons row test))
          (values (cons row train) test))))]

@bold{Building DMatrices.} Flatten the row lists into a row-major
@racket[f32vector] plus a label vector:

@chunk[<r06-build>
  (define ncol 4)
  (define (rows->dmatrix rs)
    (define n (length rs))
    (define features (make-f32vector (* n ncol)))
    (define labels (make-f32vector n))
    (for ([row (in-list rs)] [i (in-naturals)])
      (for ([v (in-list (car row))] [j (in-naturals)])
        (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
      (f32vector-set! labels i (exact->inexact (cdr row))))
    (values (make-dmatrix features #:nrow n #:ncol ncol #:labels labels) labels))]

@bold{The run.} Set up the booster with both matrices watched, iterate while
collecting @racket["mlogloss"] history, then predict on the test set, take the
argmax class per row, and tally accuracy and a 3×3 confusion matrix (indexed
@tt{truth*3 + pred}):

@chunk[<r06-run>
(define (run-example)
  (define rows (parse-iris iris-csv))
  (define-values (train-rows test-rows) (split-rows rows))
  (define-values (dtrain train-labels) (rows->dmatrix train-rows))
  (define-values (dtest  test-labels)  (rows->dmatrix test-rows))
  (define booster
    (train dtrain
           #:evals (list (cons "test" dtest))
           #:objective "multi:softprob"
           #:num-class 3
           #:eval-metric "mlogloss"
           #:max-depth 4
           #:eta 0.3
           #:verbosity 0
           #:rounds 0))
  (define eval-set (list (cons "train" dtrain) (cons "test" dtest)))
  (define history
    (for/list ([iter (in-range 50)])
      (booster-update-one-iter! booster iter dtrain)
      (parse-eval-line (eval-one-iter booster iter eval-set))))
  (define n-test (length test-rows))
  (define probs (predict booster dtest #:as 'f32vector))
  (define preds
    (for/list ([i (in-range n-test)])
      (define p0 (f32vector-ref probs (+ (* i 3) 0)))
      (define p1 (f32vector-ref probs (+ (* i 3) 1)))
      (define p2 (f32vector-ref probs (+ (* i 3) 2)))
      (cond [(and (>= p0 p1) (>= p0 p2)) 0]
            [(>= p1 p2) 1]
            [else 2])))
  (define correct
    (for/sum ([pred (in-list preds)] [i (in-range n-test)])
      (if (= pred (inexact->exact (f32vector-ref test-labels i))) 1 0)))
  (define confusion (make-vector 9 0))
  (for ([pred (in-list preds)] [i (in-range n-test)])
    (define cell (+ (* (inexact->exact (f32vector-ref test-labels i)) 3) pred))
    (vector-set! confusion cell (add1 (vector-ref confusion cell))))
  (values (/ correct n-test) confusion history n-test))]

The harness @filepath{test/06-iris.rkt} prints the training log, the test
accuracy, and the confusion matrix, and asserts the model clears a high
accuracy floor on the held-out species.

@bold{The data.} The embedded UCI Iris CSV (150 rows,
@tt{sepal-length,sepal-width,petal-length,petal-width,species}):

@chunk[<r06-data>
  (define iris-csv #<<IRIS-CSV
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica
IRIS-CSV
  )]

@chunk[<*>
  <r06-require>
  <r06-provide>
  <r06-constants>
  <r06-parse>
  <r06-split>
  <r06-build>
  <r06-run>
  <r06-data>]
