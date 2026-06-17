#lang racket/base

;; DMatrix constructors.
;;
;; `dmatrix-create-from-mat` takes a row-major f32vector directly.  The
;; array-interface variants accept XGBoost JSON __array_interface__ strings,
;; and the dense / csr / csc / columnar constructors encode Racket vectors
;; into those strings before handing them to XGBoost.

(require ffi/vector
         racket/string
         "../array-interface.rkt"
         "../error.rkt"
         "../raw/dmatrix.rkt"
         "../raw/global.rkt"
         "../structs.rkt")

(provide dmatrix-create-from-mat
         dmatrix-create-from-uri
         dmatrix-create-from-dense-array-interface
         dmatrix-create-from-csr-array-interface
         dmatrix-create-from-csc-array-interface
         dmatrix-create-from-columnar-array-interface
         dmatrix-create-from-dense
         dmatrix-create-from-csr
         dmatrix-create-from-csc
         dmatrix-create-from-columnar)

(define (dmatrix-create-from-mat data nrow ncol [missing +nan.0])
  (define expected (* nrow ncol))
  (unless (= (f32vector-length data) expected)
    (raise-argument-error 'dmatrix-create-from-mat
                          (format "f32vector of length ~a (nrow*ncol)" expected)
                          data))
  (define h
    (xgb-dmatrix-create-from-mat/raw data nrow ncol missing))
  (unless h
    (error 'dmatrix-create-from-mat
           "XGBoost returned NULL handle: ~a"
           (xgb-last-error/raw)))
  ;; Trust the user-supplied dimensions; avoids a redundant C round-trip.
  (dmatrix-impl h nrow ncol))

(define (dmatrix-create-from-uri config)
  (wrap-dmatrix
   (check-handle 'dmatrix-create-from-uri
                 (xgb-dmatrix-create-from-uri/raw config))))

(define (dmatrix-create-from-dense-array-interface data-json [config "{\"missing\":NaN}"])
  (wrap-dmatrix
   (check-handle 'dmatrix-create-from-dense-array-interface
                 (xgb-dmatrix-create-from-dense/raw data-json config))))

(define (dmatrix-create-from-csr-array-interface indptr-json indices-json data-json ncol
                                                 [config "{\"missing\":NaN}"])
  (wrap-dmatrix
   (check-handle 'dmatrix-create-from-csr-array-interface
                 (xgb-dmatrix-create-from-csr/raw indptr-json indices-json
                                                  data-json ncol config))))

(define (dmatrix-create-from-csc-array-interface indptr-json indices-json data-json nrow
                                                 [config "{\"missing\":NaN}"])
  (wrap-dmatrix
   (check-handle 'dmatrix-create-from-csc-array-interface
                 (xgb-dmatrix-create-from-csc/raw indptr-json indices-json
                                                  data-json nrow config))))

(define (dmatrix-create-from-columnar-array-interface data-json [config "{\"missing\":NaN}"])
  (wrap-dmatrix
   (check-handle 'dmatrix-create-from-columnar-array-interface
                 (xgb-dmatrix-create-from-columnar/raw data-json config))))

(define (dmatrix-create-from-dense data nrow ncol [missing +nan.0])
  (unless (= (f32vector-length data) (* nrow ncol))
    (raise-argument-error 'dmatrix-create-from-dense
                          (format "f32vector of length ~a (nrow*ncol)"
                                  (* nrow ncol))
                          data))
  (dmatrix-create-from-dense-array-interface
   (f32-array-interface data (list nrow ncol))
   (missing-config missing)))

(define (dmatrix-create-from-csr indptr indices data ncol [missing +nan.0])
  (unless (= (u32vector-length indices) (f32vector-length data))
    (error 'dmatrix-create-from-csr
           "indices length ~a does not match data length ~a"
           (u32vector-length indices)
           (f32vector-length data)))
  (unless (= (u64vector-ref indptr (sub1 (u64vector-length indptr)))
             (f32vector-length data))
    (error 'dmatrix-create-from-csr
           "final indptr value must equal data length"))
  (dmatrix-create-from-csr-array-interface
   (u64-array-interface indptr)
   (u32-array-interface indices)
   (f32-array-interface data (list (f32vector-length data)))
   ncol
   (missing-config missing)))

(define (dmatrix-create-from-csc indptr indices data nrow [missing +nan.0])
  (unless (= (u32vector-length indices) (f32vector-length data))
    (error 'dmatrix-create-from-csc
           "indices length ~a does not match data length ~a"
           (u32vector-length indices)
           (f32vector-length data)))
  (unless (= (u64vector-ref indptr (sub1 (u64vector-length indptr)))
             (f32vector-length data))
    (error 'dmatrix-create-from-csc
           "final indptr value must equal data length"))
  (dmatrix-create-from-csc-array-interface
   (u64-array-interface indptr)
   (u32-array-interface indices)
   (f32-array-interface data (list (f32vector-length data)))
   nrow
   (missing-config missing)))

(define (dmatrix-create-from-columnar columns [missing +nan.0])
  (when (null? columns)
    (raise-argument-error 'dmatrix-create-from-columnar
                          "non-empty list of f32vectors"
                          columns))
  (define nrow (f32vector-length (car columns)))
  (for ([col (in-list columns)])
    (unless (= (f32vector-length col) nrow)
      (error 'dmatrix-create-from-columnar
             "all columns must have the same length")))
  (define data-json
    (format "[~a]"
            (string-join
             (for/list ([col (in-list columns)])
               (f32-array-interface col (list nrow)))
             ",")))
  (dmatrix-create-from-columnar-array-interface data-json
                                                (missing-config missing)))
