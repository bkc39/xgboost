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
         _xgb-log-callback
         xgb-build-info/raw
         xgb-get-global-config/raw
         xgb-set-global-config/raw
         xgb-register-log-callback/raw
         _DMatrix
         _DMatrix/null
         DMatrix?
         xgb-dmatrix-free/raw
         xgb-dmatrix-create-from-mat/raw
         xgb-dmatrix-create-from-uri/raw
         xgb-dmatrix-create-from-dense/raw
         xgb-dmatrix-create-from-csr/raw
         xgb-dmatrix-create-from-csc/raw
         xgb-dmatrix-create-from-columnar/raw
         xgb-dmatrix-set-float-info/raw
         xgb-dmatrix-num-row/raw
         xgb-dmatrix-num-col/raw
         xgb-dmatrix-get-float-info/raw
         xgb-dmatrix-num-non-missing/raw
         xgb-dmatrix-get-data-as-csr/raw
         _Booster
         _Booster/null
         Booster?
         xgb-booster-free/raw
         xgb-booster-create/raw
         xgb-booster-set-param/raw
         xgb-booster-update-one-iter/raw
         xgb-booster-predict/raw
         xgb-booster-save-model/raw
         xgb-booster-load-model/raw
         xgb-booster-save-model-to-buffer/raw
         xgb-booster-load-model-from-buffer/raw
         xgb-booster-eval-one-iter/raw)

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

(define _xgb-log-callback
  (_fun _string/utf-8 -> _void))

(define-xgb xgb-build-info/raw
  (_fun (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_build_info)

(define-xgb xgb-get-global-config/raw
  (_fun (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_get_global_config)

(define-xgb xgb-set-global-config/raw
  (_fun _string/utf-8 -> _int)
  #:c-id xgb_set_global_config)

(define-xgb xgb-register-log-callback/raw
  (_fun _fpointer -> _int)
  #:c-id xgb_register_log_callback)

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

;; ---------------------------------------------------------------------------
;; Booster: opaque handle.  Same allocator/deallocator pattern as DMatrix.
;; The cache argument to `create` is a list of DMatrix handles passed as a
;; C array of `void*`; `(_list i _DMatrix)` does the marshalling and feeds
;; the length via the auxiliary `_size` argument.
;; ---------------------------------------------------------------------------

(define-cpointer-type _Booster)

(define-xgb xgb-booster-free/raw
  (_fun _Booster -> _void)
  #:c-id xgb_booster_free
  #:wrap (deallocator))

(define-xgb xgb-booster-create/raw
  (_fun (cache : (_list i _DMatrix))
        (_size = (length cache))
        -> _Booster/null)
  #:c-id xgb_booster_create
  #:wrap (allocator xgb-booster-free/raw))

(define-xgb xgb-booster-set-param/raw
  (_fun _Booster _string/utf-8 _string/utf-8 -> _int)
  #:c-id xgb_booster_set_param)

(define-xgb xgb-booster-update-one-iter/raw
  (_fun _Booster _int _DMatrix -> _int)
  #:c-id xgb_booster_update_one_iter)

;; predict:
;;   capacity:  number of floats writable in `buf` (callers pass
;;              (f32vector-length buf)).
;;   buf:       caller-owned scratch buffer.  Untouched on rc=2.
;;   out-len:   always written with the total predictions XGBoost produced.
;; Return values match the C contract: 0 success, 1 error, 2 buffer too small.
(define-xgb xgb-booster-predict/raw
  (_fun _Booster
        _DMatrix
        _string/utf-8
        (capacity : _uint64)
        (buf : _f32vector)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_predict)

;; Save/load model to/from a filesystem path.  XGBoost picks the format
;; from the extension (.json or .ubj recommended).
(define-xgb xgb-booster-save-model/raw
  (_fun _Booster _path -> _int)
  #:c-id xgb_booster_save_model)

(define-xgb xgb-booster-load-model/raw
  (_fun _Booster _path -> _int)
  #:c-id xgb_booster_load_model)

;; save_model_to_buffer mirrors predict's "size-or-fill" contract.  Caller
;; allocates `buf` of `capacity` bytes; on rc=2, `out-len` is the required
;; capacity and nothing was copied.
(define-xgb xgb-booster-save-model-to-buffer/raw
  (_fun _Booster
        _string/utf-8
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_save_model_to_buffer)

;; load_model_from_buffer takes a (data, len) pair.  We pass the bytes
;; pointer and length explicitly so Racket bytes objects round-trip without
;; assuming null-termination.
(define-xgb xgb-booster-load-model-from-buffer/raw
  (_fun _Booster
        (buf : _bytes)
        (_uint64 = (bytes-length buf))
        -> _int)
  #:c-id xgb_booster_load_model_from_buffer)

;; XGBoosterEvalOneIter takes parallel arrays of DMatrix handles + names.
;; The C shim copies the result into `buf`; rc=2 / out-len communicates
;; the required size, mirroring predict + save_model_to_buffer.
(define-xgb xgb-booster-eval-one-iter/raw
  (_fun _Booster
        (iter : _int)
        (dmats : (_list i _DMatrix))
        (names : (_list i _string/utf-8))
        (_size = (length dmats))
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_eval_one_iter)
