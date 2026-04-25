#lang racket/base

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/unsafe/define
         ffi/vector
         racket/runtime-path)

(provide xgb-version/raw
         xgb-last-error/raw
         xgb-run-regression-demo/raw
         xgb-run-classification-demo/raw
         _DMatrix
         _DMatrix/null
         DMatrix?
         xgb-dmatrix-free/raw
         xgb-dmatrix-create-from-mat/raw
         xgb-dmatrix-set-float-info/raw
         xgb-dmatrix-num-row/raw
         xgb-dmatrix-num-col/raw
         xgb-dmatrix-get-float-info/raw
         xgb-dmatrix-num-non-missing/raw
         xgb-dmatrix-get-data-as-csr/raw)

(define-runtime-path native-libs-dir "../native-libs")

(define-ffi-definer define-xgb
  (ffi-lib (build-path native-libs-dir "libxgbcompat")))

(define-xgb xgb-version/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_version)

(define-xgb xgb-last-error/raw
  (_fun -> _string/utf-8)
  #:c-id xgb_last_error)

(define-xgb xgb-run-regression-demo/raw
  (_fun (out : (_ptr o _double)) -> (rc : _int) -> (values rc out))
  #:c-id xgb_run_regression_demo)

(define-xgb xgb-run-classification-demo/raw
  (_fun (out : (_ptr o _double)) -> (rc : _int) -> (values rc out))
  #:c-id xgb_run_classification_demo)

;; ---------------------------------------------------------------------------
;; DMatrix: opaque handle to an XGBoost data matrix.
;;
;; `define-cpointer-type` generates three names:
;;   _DMatrix       — non-null cpointer type
;;   _DMatrix/null  — nullable cpointer type (used for NULL-on-error returns)
;;   DMatrix?       — predicate
;;
;; The deallocator must be defined before the allocator references it.
;; ---------------------------------------------------------------------------

(define-cpointer-type _DMatrix)

(define-xgb xgb-dmatrix-free/raw
  (_fun _DMatrix -> _void)
  #:c-id xgb_dmatrix_free
  #:wrap (deallocator))

(define-xgb xgb-dmatrix-create-from-mat/raw
  (_fun (data : _f32vector)
        (nrow : _size)
        (ncol : _size)
        (missing : _float)
        -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_mat
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-set-float-info/raw
  (_fun _DMatrix
        _string/utf-8
        (vals : _f32vector)
        (_size = (f32vector-length vals))
        -> _int)
  #:c-id xgb_dmatrix_set_float_info)

(define-xgb xgb-dmatrix-num-row/raw
  (_fun _DMatrix (out : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id xgb_dmatrix_num_row)

(define-xgb xgb-dmatrix-num-col/raw
  (_fun _DMatrix (out : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id xgb_dmatrix_num_col)

;; get_float_info writes a borrowed pointer into `out-ptr`.  The pointer is
;; owned by the DMatrix; callers must copy out of it before mutating the
;; DMatrix or letting it be collected.  The Racket wrapper
;; `dmatrix-get-float-info` does exactly that (memcpy into a fresh
;; f32vector).
(define-xgb xgb-dmatrix-get-float-info/raw
  (_fun _DMatrix
        _string/utf-8
        (out-len : (_ptr o _uint64))
        (out-ptr : (_ptr o _pointer))
        -> (rc : _int)
        -> (values rc out-len out-ptr))
  #:c-id xgb_dmatrix_get_float_info)

(define-xgb xgb-dmatrix-num-non-missing/raw
  (_fun _DMatrix (out : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id xgb_dmatrix_num_non_missing)

;; Read the matrix back as CSR.  The caller pre-allocates all three output
;; buffers; `get-data-as-csr` writes into them in place.  See the C header for
;; sizing requirements.
(define-xgb xgb-dmatrix-get-data-as-csr/raw
  (_fun _DMatrix
        _string/utf-8
        (indptr : _u64vector)
        (indices : _u32vector)
        (data : _f32vector)
        -> _int)
  #:c-id xgb_dmatrix_get_data_as_csr)
