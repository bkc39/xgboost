#lang racket/base

;; High-level DMatrix construction and metadata.
;;
;; `make-dmatrix` accepts ordinary Racket data (nested rows or flat
;; sequences) and optional labels/weights.  The remaining constructors cover
;; CSR / CSC / columnar / URI inputs, and the metadata accessors expose
;; labels, weights, group info, and feature names/types as Racket values.

(require ffi/vector
         (prefix-in foreign: "../foreign.rkt")
         "coerce.rkt")

(provide make-dmatrix
         make-dmatrix-from-csr
         make-dmatrix-from-csc
         make-dmatrix-from-columnar
         make-dmatrix-from-uri
         dmatrix->list
         dmatrix-show
         dmatrix-slice
         dmatrix-save-binary!
         dmatrix-set-label!
         dmatrix-set-weight!
         dmatrix-set-base-margin!
         dmatrix-set-label-lower-bound!
         dmatrix-set-label-upper-bound!
         dmatrix-set-group!
         dmatrix-set-feature-names!
         dmatrix-set-feature-types!
         dmatrix-label
         dmatrix-weight
         dmatrix-base-margin
         dmatrix-group-ptr
         dmatrix-feature-names
         dmatrix-feature-types
         dmatrix-quantile-cut)

(define (make-dmatrix data
                      #:nrow [nrow #f]
                      #:ncol [ncol #f]
                      #:missing [missing +nan.0]
                      #:labels [labels #f]
                      #:weights [weights #f])
  (define-values (vec rows cols) (coerce-matrix data nrow ncol))
  (define dm (foreign:dmatrix-create-from-mat vec rows cols missing))
  (when labels
    (define label-vec (sequence->f32vector 'make-dmatrix labels))
    (unless (= (f32vector-length label-vec) rows)
      (error 'make-dmatrix "label length ~a does not match row count ~a"
             (f32vector-length label-vec) rows))
    (foreign:dmatrix-set-float-info! dm "label" label-vec))
  (when weights
    (define weight-vec (sequence->f32vector 'make-dmatrix weights))
    (unless (= (f32vector-length weight-vec) rows)
      (error 'make-dmatrix "weight length ~a does not match row count ~a"
             (f32vector-length weight-vec) rows))
    (foreign:dmatrix-set-float-info! dm "weight" weight-vec))
  dm)

(define (make-dmatrix-from-csr indptr indices data ncol [missing +nan.0])
  (foreign:dmatrix-create-from-csr indptr indices data ncol missing))

(define (make-dmatrix-from-csc indptr indices data nrow [missing +nan.0])
  (foreign:dmatrix-create-from-csc indptr indices data nrow missing))

(define (make-dmatrix-from-columnar columns [missing +nan.0])
  (foreign:dmatrix-create-from-columnar columns missing))

(define (make-dmatrix-from-uri uri-or-path
                               #:format [fmt #f]
                               #:silent? [silent? #t])
  (define base
    (cond [(path? uri-or-path) (path->string uri-or-path)]
          [else uri-or-path]))
  (define uri (if fmt (format "~a?format=~a" base fmt) base))
  (define cfg
    (format "{\"uri\":~s,\"silent\":~a}" uri (if silent? 1 0)))
  (foreign:dmatrix-create-from-uri cfg))

(define (dmatrix->list dm)
  (foreign:dmatrix->list dm))

(define (dmatrix-show dm [port (current-output-port)])
  (foreign:dmatrix-show dm port))

(define (dmatrix-slice dm indices #:allow-groups? [allow-groups? #f])
  (foreign:dmatrix-slice dm indices #:allow-groups? allow-groups?))

(define (dmatrix-save-binary! dm path #:silent? [silent? #t])
  (foreign:dmatrix-save-binary! dm path #:silent? silent?))

(define (set-float-info! who dm field xs)
  (foreign:dmatrix-set-float-info! dm field (sequence->f32vector who xs)))

(define (dmatrix-set-label! dm xs) (set-float-info! 'dmatrix-set-label! dm "label" xs))
(define (dmatrix-set-weight! dm xs) (set-float-info! 'dmatrix-set-weight! dm "weight" xs))
(define (dmatrix-set-base-margin! dm xs)
  (set-float-info! 'dmatrix-set-base-margin! dm "base_margin" xs))

(define (dmatrix-set-label-lower-bound! dm xs)
  (set-float-info! 'dmatrix-set-label-lower-bound! dm "label_lower_bound" xs))

(define (dmatrix-set-label-upper-bound! dm xs)
  (set-float-info! 'dmatrix-set-label-upper-bound! dm "label_upper_bound" xs))

(define (dmatrix-set-group! dm sizes)
  (foreign:dmatrix-set-uint-info! dm "group"
                                  (sequence->u32vector 'dmatrix-set-group! sizes)))

(define (dmatrix-set-feature-names! dm names)
  (foreign:dmatrix-set-feature-info! dm "feature_name" names))

(define (dmatrix-set-feature-types! dm types)
  (foreign:dmatrix-set-feature-info! dm "feature_type" types))

(define (get-float-info-list dm field)
  (f32vector->list (foreign:dmatrix-get-float-info dm field)))

(define (dmatrix-label dm) (get-float-info-list dm "label"))
(define (dmatrix-weight dm) (get-float-info-list dm "weight"))
(define (dmatrix-base-margin dm) (get-float-info-list dm "base_margin"))

(define (dmatrix-group-ptr dm)
  (define vec (foreign:dmatrix-get-uint-info dm "group_ptr"))
  (for/list ([i (in-range (u32vector-length vec))])
    (u32vector-ref vec i)))

(define (dmatrix-feature-names dm)
  (foreign:dmatrix-get-feature-info dm "feature_name"))

(define (dmatrix-feature-types dm)
  (foreign:dmatrix-get-feature-info dm "feature_type"))

(define (dmatrix-quantile-cut dm)
  (define-values (indptr-json data-json)
    (foreign:dmatrix-get-quantile-cut dm))
  (define indptr (foreign:array-interface->u64vector indptr-json))
  (define data (foreign:array-interface->f32vector data-json))
  (values (for/list ([i (in-range (u64vector-length indptr))])
            (u64vector-ref indptr i))
          data))
