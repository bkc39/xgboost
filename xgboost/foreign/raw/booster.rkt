#lang racket/base

;; Booster: opaque handle.  Same allocator/deallocator pattern as DMatrix.
;; The cache argument to `create` is a list of DMatrix handles passed as a
;; C array of `void*`; `(_list i _DMatrix)` does the marshalling and feeds
;; the length via the auxiliary `_size` argument.

(require ffi/unsafe
         ffi/unsafe/alloc
         ffi/vector
         "library.rkt"
         "dmatrix.rkt")

(provide _Booster
         _Booster/null
         Booster?
         xgb-booster-free/raw
         xgb-booster-create/raw
         xgb-booster-reset/raw
         xgb-booster-slice/raw
         xgb-booster-boosted-rounds/raw
         xgb-booster-num-feature/raw
         xgb-booster-set-param/raw
         xgb-booster-update-one-iter/raw
         xgb-booster-train-one-iter/raw
         xgb-booster-predict/raw
         xgb-booster-predict-from-dense/raw
         xgb-booster-predict-from-csr/raw
         xgb-booster-predict-from-columnar/raw
         xgb-booster-save-model/raw
         xgb-booster-load-model/raw
         xgb-booster-save-model-to-buffer/raw
         xgb-booster-load-model-from-buffer/raw
         xgb-booster-serialize-to-buffer/raw
         xgb-booster-unserialize-from-buffer/raw
         xgb-booster-save-json-config/raw
         xgb-booster-load-json-config/raw
         xgb-booster-set-attr/raw
         xgb-booster-delete-attr/raw
         xgb-booster-get-attr/raw
         xgb-booster-get-attr-names/raw
         xgb-booster-set-str-feature-info/raw
         xgb-booster-get-str-feature-info/raw
         xgb-booster-dump-model/raw
         xgb-booster-dump-model-with-features/raw
         xgb-booster-feature-score/raw
         xgb-booster-eval-one-iter/raw)

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

(define-xgb xgb-booster-reset/raw
  (_fun _Booster -> _int)
  #:c-id xgb_booster_reset)

(define-xgb xgb-booster-slice/raw
  (_fun _Booster _int _int _int -> _Booster/null)
  #:c-id xgb_booster_slice
  #:wrap (allocator xgb-booster-free/raw))

(define-xgb xgb-booster-boosted-rounds/raw
  (_fun _Booster (out : (_ptr o _int))
        -> (rc : _int)
        -> (values rc out))
  #:c-id xgb_booster_boosted_rounds)

(define-xgb xgb-booster-num-feature/raw
  (_fun _Booster (out : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out))
  #:c-id xgb_booster_num_feature)

(define-xgb xgb-booster-set-param/raw
  (_fun _Booster _string/utf-8 _string/utf-8 -> _int)
  #:c-id xgb_booster_set_param)

(define-xgb xgb-booster-update-one-iter/raw
  (_fun _Booster _int _DMatrix -> _int)
  #:c-id xgb_booster_update_one_iter)

(define-xgb xgb-booster-train-one-iter/raw
  (_fun _Booster _DMatrix _int _string/utf-8 _string/utf-8 -> _int)
  #:c-id xgb_booster_train_one_iter)

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

(define-xgb xgb-booster-predict-from-dense/raw
  (_fun _Booster
        _string/utf-8
        _string/utf-8
        _DMatrix/null
        (capacity : _uint64)
        (buf : _f32vector)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_predict_from_dense)

(define-xgb xgb-booster-predict-from-csr/raw
  (_fun _Booster
        _string/utf-8
        _string/utf-8
        _string/utf-8
        _uint64
        _string/utf-8
        _DMatrix/null
        (capacity : _uint64)
        (buf : _f32vector)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_predict_from_csr)

(define-xgb xgb-booster-predict-from-columnar/raw
  (_fun _Booster
        _string/utf-8
        _string/utf-8
        _DMatrix/null
        (capacity : _uint64)
        (buf : _f32vector)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_predict_from_columnar)

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

;; serialize_to_buffer captures full booster state (training caches +
;; iteration counters), so a snapshot taken mid-training can resume
;; update_one_iter calls in a fresh booster.  Same size-then-fill contract
;; as save_model_to_buffer.
(define-xgb xgb-booster-serialize-to-buffer/raw
  (_fun _Booster
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_serialize_to_buffer)

(define-xgb xgb-booster-unserialize-from-buffer/raw
  (_fun _Booster
        (buf : _bytes)
        (_uint64 = (bytes-length buf))
        -> _int)
  #:c-id xgb_booster_unserialize_from_buffer)

(define-xgb xgb-booster-save-json-config/raw
  (_fun _Booster
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len))
  #:c-id xgb_booster_save_json_config)

(define-xgb xgb-booster-load-json-config/raw
  (_fun _Booster _string/utf-8 -> _int)
  #:c-id xgb_booster_load_json_config)

(define-xgb xgb-booster-set-attr/raw
  (_fun _Booster _string/utf-8 _string/utf-8 -> _int)
  #:c-id xgb_booster_set_attr)

(define-xgb xgb-booster-delete-attr/raw
  (_fun _Booster _string/utf-8 -> _int)
  #:c-id xgb_booster_delete_attr)

(define-xgb xgb-booster-get-attr/raw
  (_fun _Booster
        _string/utf-8
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (found : (_ptr o _int))
        -> (rc : _int)
        -> (values rc out-len found))
  #:c-id xgb_booster_get_attr)

(define-xgb xgb-booster-get-attr-names/raw
  (_fun _Booster
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (out-count : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len out-count))
  #:c-id xgb_booster_get_attr_names)

(define-xgb xgb-booster-set-str-feature-info/raw
  (_fun _Booster
        _string/utf-8
        (vals : (_list i _string/utf-8))
        (_uint64 = (length vals))
        -> _int)
  #:c-id xgb_booster_set_str_feature_info)

(define-xgb xgb-booster-get-str-feature-info/raw
  (_fun _Booster
        _string/utf-8
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (out-count : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len out-count))
  #:c-id xgb_booster_get_str_feature_info)

(define-xgb xgb-booster-dump-model/raw
  (_fun _Booster
        _string/utf-8
        _int
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (out-count : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len out-count))
  #:c-id xgb_booster_dump_model)

(define-xgb xgb-booster-dump-model-with-features/raw
  (_fun _Booster
        (feature-names : (_list i _string/utf-8))
        (feature-types : (_list i _string/utf-8))
        (_uint64 = (length feature-names))
        _string/utf-8
        _int
        (capacity : _uint64)
        (buf : _bytes)
        (out-len : (_ptr o _uint64))
        (out-count : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-len out-count))
  #:c-id xgb_booster_dump_model_with_features)

(define-xgb xgb-booster-feature-score/raw
  (_fun _Booster
        _string/utf-8
        (feature-capacity : _uint64)
        (feature-buf : _bytes)
        (out-feature-len : (_ptr o _uint64))
        (out-n-features : (_ptr o _uint64))
        (shape-capacity : _uint64)
        (shape-buf : _u64vector)
        (out-dim : (_ptr o _uint64))
        (score-capacity : _uint64)
        (score-buf : _f32vector)
        (out-n-scores : (_ptr o _uint64))
        -> (rc : _int)
        -> (values rc out-feature-len out-n-features out-dim out-n-scores))
  #:c-id xgb_booster_feature_score)

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
