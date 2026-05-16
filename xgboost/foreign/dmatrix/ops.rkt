#lang racket/base

;; DMatrix operations: shape queries, row slicing, binary save, quantile-cut
;; inspection, dense reconstruction (`dmatrix->list`), and the polars-style
;; `dmatrix-show` pretty-printer.

(require ffi/vector
         racket/string
         "../error.rkt"
         "../raw/dmatrix.rkt"
         "../raw/global.rkt"
         "../structs.rkt"
         "metadata.rkt")

(provide dmatrix-slice
         dmatrix-nrow
         dmatrix-ncol
         dmatrix-num-non-missing
         dmatrix-save-binary!
         dmatrix-get-quantile-cut
         dmatrix->list
         dmatrix-show)

(define (dmatrix-nrow dm) (dmatrix-rows dm))
(define (dmatrix-ncol dm) (dmatrix-cols dm))

(define (sequence->s32vector who xs)
  (define vals
    (cond
      [(s32vector? xs) #f]
      [(list? xs) xs]
      [(vector? xs) (vector->list xs)]
      [else (raise-argument-error who "list, vector, or s32vector" xs)]))
  (cond
    [(s32vector? xs) xs]
    [else
     (for ([v (in-list vals)])
       (unless (and (exact-integer? v) (>= v 0))
         (raise-argument-error who "nonnegative exact integer row index" v)))
     (list->s32vector vals)]))

(define (dmatrix-slice h rows #:allow-groups? [allow-groups? #f])
  (define idx (sequence->s32vector 'dmatrix-slice rows))
  (define sliced
    (xgb-dmatrix-slice/raw h idx (if allow-groups? 1 0)))
  (unless sliced
    (error 'dmatrix-slice
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  (wrap-dmatrix sliced))

(define (dmatrix-num-non-missing h)
  (define-values (rc out) (xgb-dmatrix-num-non-missing/raw h))
  (check-ok rc 'dmatrix-num-non-missing)
  out)

(define (dmatrix-save-binary! h path #:silent? [silent? #t])
  (check-ok (xgb-dmatrix-save-binary/raw h path (if silent? 1 0))
            'dmatrix-save-binary!))

(define (dmatrix-get-quantile-cut h [config "{}"])
  (define-values (rc indptr-len data-len)
    (xgb-dmatrix-get-quantile-cut/raw h config
                                      0 (make-bytes 0)
                                      0 (make-bytes 0)))
  (cond
    [(zero? rc) (values "" "")]
    [(= rc 2)
     (define indptr-buf (make-bytes indptr-len))
     (define data-buf (make-bytes data-len))
     (define-values (rc2 indptr-len2 data-len2)
       (xgb-dmatrix-get-quantile-cut/raw h config
                                         indptr-len indptr-buf
                                         data-len data-buf))
     (check-ok rc2 'dmatrix-get-quantile-cut)
     (unless (and (= indptr-len2 indptr-len)
                  (= data-len2 data-len))
       (error 'dmatrix-get-quantile-cut
              "expected ~a/~a bytes, got ~a/~a"
              indptr-len data-len indptr-len2 data-len2))
     (values (bytes->string/utf-8 indptr-buf #f 0 indptr-len)
             (bytes->string/utf-8 data-buf #f 0 data-len))]
    [else (check-ok rc 'dmatrix-get-quantile-cut)]))

;; Reconstruct a dense (nrow x ncol) list-of-lists from the DMatrix's CSR
;; storage.  Missing entries (absent in the CSR) materialize as `+nan.0`;
;; when the DMatrix was built with no missing values, nnz == nrow*ncol and
;; every slot is filled.
(define (dmatrix->list h)
  (define nrow (dmatrix-nrow h))
  (define ncol (dmatrix-ncol h))
  (define nnz (dmatrix-num-non-missing h))
  (define indptr (make-u64vector (+ nrow 1)))
  (define indices (make-u32vector nnz))
  (define data (make-f32vector nnz))
  (check-ok (xgb-dmatrix-get-data-as-csr/raw h "{}" indptr indices data)
            'dmatrix->list)
  (for/list ([r (in-range nrow)])
    (define row (make-vector ncol +nan.0))
    (for ([k (in-range (u64vector-ref indptr r)
                       (u64vector-ref indptr (+ r 1)))])
      (vector-set! row (u32vector-ref indices k) (f32vector-ref data k)))
    (vector->list row)))

;; List of float info fields we probe when showing a DMatrix.  XGBoost
;; returns an empty vector for unset fields, so trying them all is cheap.
(define dmatrix-known-float-info-fields
  '("label" "weight" "base_margin" "label_lower_bound" "label_upper_bound"))

(define dmatrix-show-preview-head 8)
(define dmatrix-show-preview-tail 4)

(define (f32vector->preview-string vec)
  (define n (f32vector-length vec))
  (define (el i) (number->string (f32vector-ref vec i)))
  (cond
    [(<= n (+ dmatrix-show-preview-head dmatrix-show-preview-tail))
     (string-join (for/list ([i (in-range n)]) (el i)) " ")]
    [else
     (define head
       (string-join (for/list ([i (in-range dmatrix-show-preview-head)])
                      (el i))
                    " "))
     (define tail
       (string-join (for/list ([i (in-range (- n dmatrix-show-preview-tail)
                                            n)])
                      (el i))
                    " "))
     (format "~a ... ~a  (len=~a)" head tail n)]))

(define dmatrix-show-max-rows 4)
(define dmatrix-show-max-cols 4)

(define (format-cell v)
  (cond
    [(not (= v v)) "NaN"]              ; NaN is the only value != itself
    [else (real->decimal-string v 4)]))

(define (pad-left s width)
  (define pad (- width (string-length s)))
  (if (<= pad 0) s (string-append (make-string pad #\space) s)))

(define (dmatrix-show h [port (current-output-port)])
  (define nrow (dmatrix-nrow h))
  (define ncol (dmatrix-ncol h))
  (fprintf port "DMatrix ~ax~a\n" nrow ncol)
  (for ([field (in-list dmatrix-known-float-info-fields)])
    (define vals (dmatrix-get-float-info h field))
    (when (> (f32vector-length vals) 0)
      (fprintf port "  ~a: [~a]\n" field (f32vector->preview-string vals))))
  (when (and (> nrow 0) (> ncol 0))
    (define row-cutoff (min nrow dmatrix-show-max-rows))
    (define col-cutoff (min ncol dmatrix-show-max-cols))
    (define cols-truncated? (> ncol col-cutoff))
    (define rows-truncated? (> nrow row-cutoff))
    (define row-tail (if cols-truncated? " ..." ""))
    ;; Two-pass: format every cell in the preview window, then right-align to
    ;; the max width so columns line up like a polars dataframe preview.
    (define formatted-rows
      (for/list ([row (in-list (dmatrix->list h))]
                 [_r (in-range row-cutoff)])
        (for/list ([v (in-list row)]
                   [_c (in-range col-cutoff)])
          (format-cell v))))
    (define col-width
      (for*/fold ([w 0])
                 ([row (in-list formatted-rows)]
                  [cell (in-list row)])
        (max w (string-length cell))))
    (for ([row (in-list formatted-rows)])
      (define cells (for/list ([cell (in-list row)]) (pad-left cell col-width)))
      (fprintf port "  ~a~a\n" (string-join cells " ") row-tail))
    (when rows-truncated?
      (fprintf port "  ...\n"))))
