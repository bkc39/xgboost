#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/file
                     xgboost))

@section[#:tag "ex-save-load"]{Saving and loading models}

A trained model can be persisted two ways: to a file with @racket[save-model] /
@racket[load-model], or to an in-memory bytes blob with
@racket[save-model-to-bytes] / @racket[load-model-from-bytes] (in either UBJ or
JSON format). This example trains a regressor, round-trips it through all three,
and confirms every reloaded booster reproduces the original predictions
bit-for-bit.

@chunk[<r03-require>
(require ffi/vector
         racket/file
         xgboost)]

@chunk[<r03-provide>
(provide run-example)]

@bold{Helpers.} A trained booster on the usual eight-row set, and an exact
@racket[f32vector] equality check:

@chunk[<r03-helpers>
  (define (vec=? a b)
    (and (= (f32vector-length a) (f32vector-length b))
         (for/and ([i (in-range (f32vector-length a))])
           (= (f32vector-ref a i) (f32vector-ref b i)))))]

@bold{The run.} Save to a temp file and to UBJ/JSON byte blobs, reload each, and
compare predictions to the baseline. @racket[run-example] returns the blob sizes
and whether each reload matched:

@chunk[<r03-run>
(define (run-example)
  (define dtrain
    (make-dmatrix (f32vector 1.0 2.0 0.5   2.0 1.0 1.5   3.0 0.5 0.0   0.5 3.0 2.0
                             4.0 2.0 1.0   1.5 1.5 0.5   2.5 3.5 1.5   0.0 1.0 0.0)
                  #:nrow 8 #:ncol 3
                  #:labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
  (define booster
    (train dtrain #:objective "reg:squarederror"
           #:max-depth 3 #:eta 0.1 #:verbosity 0 #:rounds 50))
  (define baseline (predict booster dtrain #:as 'f32vector))
  ;; File round-trip.
  (define model-path (make-temporary-file "xgbrkt-~a.json"))
  (save-model booster model-path)
  (define file-bytes (file->bytes model-path))
  (define preds-file (predict (load-model model-path) dtrain #:as 'f32vector))
  (delete-file model-path)
  ;; Byte-buffer round-trips (UBJ default, plus JSON).
  (define ubj-blob (save-model-to-bytes booster))
  (define json-blob (save-model-to-bytes booster #:format "json"))
  (define preds-ubj (predict (load-model-from-bytes ubj-blob) dtrain #:as 'f32vector))
  (define preds-json (predict (load-model-from-bytes json-blob) dtrain #:as 'f32vector))
  (hash 'file-bytes (bytes-length file-bytes)
        'ubj-bytes (bytes-length ubj-blob)
        'json-bytes (bytes-length json-blob)
        'file-match? (vec=? preds-file baseline)
        'ubj-match? (vec=? preds-ubj baseline)
        'json-match? (vec=? preds-json baseline)
        'baseline (f32vector->list baseline)))]

The harness @filepath{test/03-save-load.rkt} prints the blob sizes and a
match table, and asserts every reloaded booster reproduces the baseline exactly.

@chunk[<*>
  <r03-require>
  <r03-provide>
  <r03-helpers>
  <r03-run>]
