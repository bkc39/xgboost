#lang scribble/lp2

@(require (for-label ffi/vector
                     racket/base
                     racket/file
                     xgboost))

@section[#:tag "ex-dmatrix-constructors"]{DMatrix constructors}

Beyond the dense @racket[make-dmatrix], the bindings offer constructors for the
common sparse and external layouts: CSR (@racket[make-dmatrix-from-csr]), CSC
(@racket[make-dmatrix-from-csc]), columnar struct-of-arrays
(@racket[make-dmatrix-from-columnar]), and libsvm/CSV files
(@racket[make-dmatrix-from-uri]). This example builds the same little @tt{2×3}
matrix five different ways and reads each back with @racket[dmatrix->list].

@chunk[<r12-require>
(require ffi/vector
         racket/file
         xgboost)]

@chunk[<r12-provide>
(provide run-example)]

@bold{Helpers.} A per-matrix summary, and a libsvm file written to a temp path
then loaded by URI:

@chunk[<r12-helpers>
  (define (matrix-summary dm)
    (hash 'rows (dmatrix-rows dm) 'cols (dmatrix-cols dm)
          'values (dmatrix->list dm)))
  (define (make-libsvm-dmatrix)
    (define tmp (make-temporary-file "xgboost-e2e-~a.libsvm"))
    (dynamic-wind
      void
      (lambda ()
        (call-with-output-file tmp
          (lambda (out)
            (displayln "0 0:1 1:2 2:3" out)
            (displayln "1 0:4 1:5 2:6" out))
          #:exists 'truncate)
        (make-dmatrix-from-uri tmp #:format "libsvm"))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))]

@bold{The run.} The dense and libsvm forms carry all six values; the CSR/CSC
forms omit the two zero entries (which read back as @racket[+nan.0]).
@racket[run-example] returns a summary of each:

@chunk[<r12-run>
(define (run-example)
  (define dense
    (make-dmatrix (f32vector 1.0 2.0 3.0  4.0 5.0 6.0)
                  #:nrow 2 #:ncol 3 #:missing -1.0))
  (define csr
    (make-dmatrix-from-csr (u64vector 0 2 4) (u32vector 0 2 1 2)
                           (f32vector 1.0 3.0 5.0 6.0) 3 -1.0))
  (define csc
    (make-dmatrix-from-csc (u64vector 0 1 2 4) (u32vector 0 1 0 1)
                           (f32vector 1.0 5.0 3.0 6.0) 2 -1.0))
  (define columnar
    (make-dmatrix-from-columnar
     (list (f32vector 1.0 4.0) (f32vector 2.0 5.0) (f32vector 3.0 6.0)) -1.0))
  (hash 'dense (matrix-summary dense)
        'csr (matrix-summary csr)
        'csc (matrix-summary csc)
        'columnar (matrix-summary columnar)
        'libsvm (matrix-summary (make-libsvm-dmatrix))))]

The harness @filepath{test/12-dmatrix-constructors.rkt} prints each matrix's
shape and asserts the dense/columnar/libsvm forms match the full matrix while
CSR/CSC reproduce the sparse one.

@chunk[<*>
  <r12-require>
  <r12-provide>
  <r12-helpers>
  <r12-run>]
