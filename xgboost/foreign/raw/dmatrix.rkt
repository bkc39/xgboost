#lang racket/base

;; DMatrix: opaque handle to an XGBoost data matrix.
;;
;; `define-cpointer-type` generates three names:
;;   _DMatrix       — non-null cpointer type
;;   _DMatrix/null  — nullable cpointer type (used for NULL-on-error returns)
;;   DMatrix?       — predicate
;;
;; The deallocator must be defined before the allocator references it.

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/vector
         "library.rkt")

(provide _DMatrix
         _DMatrix/null ;; noqa
         DMatrix? ;; noqa
         xgb-dmatrix-free/raw
         xgb-dmatrix-create-from-mat/raw
         xgb-dmatrix-create-from-uri/raw
         xgb-dmatrix-create-from-dense/raw
         xgb-dmatrix-create-from-csr/raw
         xgb-dmatrix-create-from-csc/raw
         xgb-dmatrix-create-from-columnar/raw
         xgb-dmatrix-slice/raw
         xgb-dmatrix-set-float-info/raw
         xgb-dmatrix-set-uint-info/raw
         xgb-dmatrix-set-info-from-interface/raw
         xgb-dmatrix-set-str-feature-info/raw
         xgb-dmatrix-get-str-feature-info/raw
         xgb-dmatrix-num-row/raw
         xgb-dmatrix-num-col/raw
         xgb-dmatrix-get-float-info/raw
         xgb-dmatrix-get-uint-info/raw
         xgb-dmatrix-num-non-missing/raw
         xgb-dmatrix-get-data-as-csr/raw
         xgb-dmatrix-save-binary/raw
         xgb-dmatrix-get-quantile-cut/raw)

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

(define-xgb xgb-dmatrix-create-from-uri/raw
  (_fun _string/utf-8 -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_uri
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-create-from-dense/raw
  (_fun _string/utf-8 _string/utf-8 -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_dense
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-create-from-csr/raw
  (_fun _string/utf-8
        _string/utf-8
        _string/utf-8
        _uint64
        _string/utf-8
        -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_csr
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-create-from-csc/raw
  (_fun _string/utf-8
        _string/utf-8
        _string/utf-8
        _uint64
        _string/utf-8
        -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_csc
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-create-from-columnar/raw
  (_fun _string/utf-8 _string/utf-8 -> _DMatrix/null)
  #:c-id xgb_dmatrix_create_from_columnar
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-slice/raw
  (_fun _DMatrix
        (indices : _s32vector)
        (_size = (s32vector-length indices))
        _int
        -> _DMatrix/null)
  #:c-id xgb_dmatrix_slice
  #:wrap (allocator xgb-dmatrix-free/raw))

(define-xgb xgb-dmatrix-set-float-info/raw
  (_fun _DMatrix
        _string/utf-8
        (vals : _f32vector)
        (_size = (f32vector-length vals))
        -> _int)
  #:c-id xgb_dmatrix_set_float_info)

(define-xgb xgb-dmatrix-set-uint-info/raw
  (_fun _DMatrix
        _string/utf-8
        (vals : _u32vector)
        (_size = (u32vector-length vals))
        -> _int)
  #:c-id xgb_dmatrix_set_uint_info)

(define-xgb xgb-dmatrix-set-info-from-interface/raw
  (_fun _DMatrix _string/utf-8 _string/utf-8 -> _int)
  #:c-id xgb_dmatrix_set_info_from_interface)

(define-xgb xgb-dmatrix-set-str-feature-info/raw
  (_fun _DMatrix
        _string/utf-8
        (vals : (_list i _string/utf-8))
        (_uint64 = (length vals))
        -> _int)
  #:c-id xgb_dmatrix_set_str_feature_info)

(define-xgb xgb-dmatrix-get-str-feature-info/raw
  (_fun _DMatrix
        _string/utf-8
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (out-count : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len out-count))
  #:c-id xgb_dmatrix_get_str_feature_info)

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

(define-xgb xgb-dmatrix-get-uint-info/raw
  (_fun _DMatrix
        _string/utf-8
        (out-len : (_ptr o _uint64))
        (out-ptr : (_ptr o _pointer))
        -> (rc : _int)
        -> (values rc out-len out-ptr))
  #:c-id xgb_dmatrix_get_uint_info)

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

(define-xgb xgb-dmatrix-save-binary/raw
  (_fun _DMatrix _path _int -> _int)
  #:c-id xgb_dmatrix_save_binary)

(define-xgb xgb-dmatrix-get-quantile-cut/raw
  (_fun _DMatrix
        _string/utf-8
        (indptr-capacity : _uint64)
        (indptr-buf : _bytes)
        (out-indptr-len : (_ptr o _uint64))
        (data-capacity : _uint64)
        (data-buf : _bytes)
        (out-data-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-indptr-len out-data-len))
  #:c-id xgb_dmatrix_get_quantile_cut)
