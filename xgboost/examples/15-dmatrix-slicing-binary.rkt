#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/file
                     xgboost))

@section[#:tag "ex-dmatrix-slicing-binary"]{Slicing and binary serialization}

@racket[dmatrix-slice] selects a subset of rows (by index, in the order given),
and @racket[dmatrix-save-binary!] writes a DMatrix to XGBoost's fast binary
format that @racket[make-dmatrix-from-uri] reads back. This example slices rows
@racket['(2 0)] from a @tt{3×2} matrix --- reordering them --- then saves and
reloads the slice.

@chunk[<r15-require>
(require ffi/vector
         racket/file
         xgboost)]

@chunk[<r15-provide>
(provide run-example)]

@chunk[<r15-run>
(define (run-example)
  (define dm
    (make-dmatrix (f32vector 1.0 2.0  3.0 4.0  5.0 6.0)
                  #:nrow 3 #:ncol 2 #:missing -1.0))
  (define sliced (dmatrix-slice dm '(2 0)))
  (define sliced-values (dmatrix->list sliced))
  (define tmp (make-temporary-file "xgboost-dmatrix-~a.buffer"))
  (when (file-exists? tmp) (delete-file tmp))
  (define loaded-summary
    (dynamic-wind
      void
      (lambda ()
        (dmatrix-save-binary! sliced tmp)
        (define loaded (make-dmatrix-from-uri tmp))
        (hash 'rows (dmatrix-rows loaded) 'cols (dmatrix-cols loaded)
              'values (dmatrix->list loaded)))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))
  (hash 'sliced-values sliced-values 'loaded-summary loaded-summary))]

The harness @filepath{test/15-dmatrix-slicing-binary.rkt} prints the sliced
values and the reloaded shape, and asserts both equal the expected reordered
rows @racket['((5.0 6.0) (1.0 2.0))].

@chunk[<*>
  <r15-require>
  <r15-provide>
  <r15-run>]
