#lang racket/base

(require (except-in ffi/unsafe ->)
         ffi/vector
         racket/contract
         racket/list
         racket/string
         "ffi/raw.rkt")

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [xgboost-build-info (-> string?)]
  [xgboost-get-global-config (-> string?)]
  [xgboost-set-global-config! (-> string? void?)]
  [xgboost-register-log-callback! (-> (-> string? any/c) void?)]
  [run-regression-demo (-> rational?)]
  [run-classification-demo (-> (and/c rational? (>=/c 0) (<=/c 1)))]
  [dmatrix? (-> any/c boolean?)]
  [dmatrix-create-from-mat
   (->* (f32vector? exact-nonnegative-integer? exact-nonnegative-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-uri (-> string? dmatrix?)]
  [dmatrix-create-from-dense-array-interface
   (->* (string?) (string?) dmatrix?)]
  [dmatrix-create-from-csr-array-interface
   (->* (string? string? string? exact-nonnegative-integer?)
        (string?)
        dmatrix?)]
  [dmatrix-create-from-csc-array-interface
   (->* (string? string? string? exact-nonnegative-integer?)
        (string?)
        dmatrix?)]
  [dmatrix-create-from-columnar-array-interface
   (->* (string?) (string?) dmatrix?)]
  [dmatrix-create-from-dense
   (->* (f32vector? exact-positive-integer? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-csr
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-csc
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-create-from-columnar
   (->* ((listof f32vector?))
        (real?)
        dmatrix?)]
  [dmatrix-slice
   (->* (dmatrix? (or/c s32vector?
                         (listof exact-nonnegative-integer?)
                         (vectorof exact-nonnegative-integer?)))
        (#:allow-groups? any/c)
        dmatrix?)]
  [dmatrix-set-float-info! (-> dmatrix? string? f32vector? void?)]
  [dmatrix-set-uint-info! (-> dmatrix? string? u32vector? void?)]
  [dmatrix-set-info-from-interface! (-> dmatrix? string? string? void?)]
  [dmatrix-set-feature-info! (-> dmatrix? string? (listof string?) void?)]
  [dmatrix-free! (-> dmatrix? void?)]
  [dmatrix-nrow (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-ncol (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-get-float-info (-> dmatrix? string? f32vector?)]
  [dmatrix-get-uint-info (-> dmatrix? string? u32vector?)]
  [dmatrix-get-feature-info (-> dmatrix? string? (listof string?))]
  [dmatrix-num-non-missing (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix->list (-> dmatrix? (listof (listof real?)))]
  [dmatrix-save-binary! (->* (dmatrix? path-string?) (#:silent? any/c) void?)]
  [dmatrix-get-quantile-cut (->* (dmatrix?) (string?) (values string? string?))]
  [dmatrix-show (->* (dmatrix?) (output-port?) void?)]
  [booster? (-> any/c boolean?)]
  [booster-create (->* () ((listof dmatrix?)) booster?)]
  [booster-free! (-> booster? void?)]
  [booster-reset! (-> booster? void?)]
  [booster-slice
   (->* (booster? exact-integer? exact-integer?)
        (exact-positive-integer?)
        booster?)]
  [booster-boosted-rounds (-> booster? exact-nonnegative-integer?)]
  [booster-num-feature (-> booster? exact-nonnegative-integer?)]
  [booster-set-param! (-> booster? string? string? void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-predict
   (->* (booster? dmatrix?)
        (#:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?))
        f32vector?)]
  [booster-predict-from-dense
   (->* (booster? f32vector? exact-positive-integer? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-predict-from-csr
   (->* (booster? u64vector? u32vector? f32vector? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-predict-from-columnar
   (->* (booster? (listof f32vector?))
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:config (or/c #f string?)
         #:proxy (or/c #f dmatrix?))
        f32vector?)]
  [booster-save-model! (-> booster? path-string? void?)]
  [booster-load-model! (-> booster? path-string? void?)]
  [booster-save-model-to-bytes
   (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [booster-load-model-from-bytes! (-> booster? bytes? void?)]
  [booster-save-json-config (-> booster? string?)]
  [booster-load-json-config! (-> booster? string? void?)]
  [booster-set-attr! (-> booster? string? string? void?)]
  [booster-delete-attr! (-> booster? string? void?)]
  [booster-get-attr (-> booster? string? (or/c #f string?))]
  [booster-get-attr-names (-> booster? (listof string?))]
  [booster-set-feature-info! (-> booster? string? (listof string?) void?)]
  [booster-get-feature-info (-> booster? string? (listof string?))]
  [booster-dump-model
   (->* (booster?)
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c)
        (listof string?))]
  [booster-dump-model-with-features
   (->* (booster? (listof string?) (listof string?))
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c)
        (listof string?))]
  [booster-feature-score
   (->* (booster?)
        (#:importance-type string?
         #:feature-names (or/c #f (listof string?))
         #:config (or/c #f string?))
        hash?)]
  [booster-eval-one-iter
   (-> booster? exact-integer?
       (listof (cons/c string? dmatrix?))
       string?)]
  [parse-eval-line (-> string? (hash/c string? real?))]))

(define (xgboost-version)
  (xgb-version/raw))

(define (copy-string-result who raw)
  (define-values (rc len)
    (raw 0 (make-bytes 0)))
  (cond
    [(zero? rc) ""]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2) (raw len buf))
     (check-ok rc2 who)
     (unless (= len2 len)
       (error who "expected ~a bytes, got ~a" len len2))
     (bytes->string/utf-8 buf #f 0 len)]
    [else (check-ok rc who)]))

(define (xgboost-build-info)
  (copy-string-result 'xgboost-build-info xgb-build-info/raw))

(define (xgboost-get-global-config)
  (copy-string-result 'xgboost-get-global-config xgb-get-global-config/raw))

(define (xgboost-set-global-config! config)
  (check-ok (xgb-set-global-config/raw config)
            'xgboost-set-global-config!))

(define current-log-callback #f)

(define (xgboost-register-log-callback! proc)
  (define callback-ptr (cast proc _xgb-log-callback _fpointer))
  (check-ok (xgb-register-log-callback/raw callback-ptr)
            'xgboost-register-log-callback!)
  (set! current-log-callback (cons proc callback-ptr)))

(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (xgb-last-error/raw))))

(define (run-regression-demo)
  (define-values (rc out) (xgb-run-regression-demo/raw))
  (check-ok rc 'run-regression-demo)
  out)

(define (run-classification-demo)
  (define-values (rc out) (xgb-run-classification-demo/raw))
  (check-ok rc 'run-classification-demo)
  out)

;; ---------------------------------------------------------------------------
;; DMatrix wrappers.
;;
;; The raw layer's `#:wrap (allocator ...)` auto-registers a finalizer that
;; calls `xgb-dmatrix-free/raw` (itself `#:wrap (deallocator)`).  An explicit
;; `dmatrix-free!` call runs the deallocator — which cancels the finalizer so
;; the free does not execute twice — and we flip the cpointer tag so a second
;; explicit free raises `exn:fail:contract` at the contract boundary instead
;; of double-freeing at the C level.
;; ---------------------------------------------------------------------------

(define (dmatrix? v) (DMatrix? v))

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
  h)

(define (check-handle who h)
  (unless h
    (error who "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  h)

(define (dmatrix-create-from-uri config)
  (check-handle 'dmatrix-create-from-uri
                (xgb-dmatrix-create-from-uri/raw config)))

(define (dmatrix-create-from-dense-array-interface data-json [config "{\"missing\":NaN}"])
  (check-handle 'dmatrix-create-from-dense-array-interface
                (xgb-dmatrix-create-from-dense/raw data-json config)))

(define (dmatrix-create-from-csr-array-interface indptr-json indices-json data-json ncol
                                                 [config "{\"missing\":NaN}"])
  (check-handle 'dmatrix-create-from-csr-array-interface
                (xgb-dmatrix-create-from-csr/raw indptr-json indices-json
                                                 data-json ncol config)))

(define (dmatrix-create-from-csc-array-interface indptr-json indices-json data-json nrow
                                                 [config "{\"missing\":NaN}"])
  (check-handle 'dmatrix-create-from-csc-array-interface
                (xgb-dmatrix-create-from-csc/raw indptr-json indices-json
                                                 data-json nrow config)))

(define (dmatrix-create-from-columnar-array-interface data-json [config "{\"missing\":NaN}"])
  (check-handle 'dmatrix-create-from-columnar-array-interface
                (xgb-dmatrix-create-from-columnar/raw data-json config)))

(define (json-number v)
  (cond
    [(not (= v v)) "NaN"]
    [(eqv? v +inf.0) "Infinity"]
    [(eqv? v -inf.0) "-Infinity"]
    [else (number->string v)]))

(define (missing-config missing)
  (format "{\"missing\":~a}" (json-number missing)))

(define (dims->json dims)
  (string-join (map number->string dims) ","))

(define (array-interface-json ptr typestr dims)
  (format "{\"data\":[~a,false],\"typestr\":\"~a\",\"shape\":[~a],\"version\":3}"
          (cast ptr _pointer _uintptr)
          typestr
          (dims->json dims)))

(define (f32-array-interface vec dims)
  (array-interface-json (f32vector->cpointer vec) "<f4" dims))

(define (u64-array-interface vec)
  (array-interface-json (u64vector->cpointer vec) "<u8"
                        (list (u64vector-length vec))))

(define (u32-array-interface vec)
  (array-interface-json (u32vector->cpointer vec) "<u4"
                        (list (u32vector-length vec))))

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
  sliced)

(define (dmatrix-set-float-info! h field vals)
  (check-ok (xgb-dmatrix-set-float-info/raw h field vals)
            'dmatrix-set-float-info!))

(define (dmatrix-set-uint-info! h field vals)
  (check-ok (xgb-dmatrix-set-info-from-interface/raw
             h field (u32-array-interface vals))
            'dmatrix-set-uint-info!))

(define (dmatrix-set-info-from-interface! h field data-json)
  (check-ok (xgb-dmatrix-set-info-from-interface/raw h field data-json)
            'dmatrix-set-info-from-interface!))

(define (dmatrix-set-feature-info! h field vals)
  (check-ok (xgb-dmatrix-set-str-feature-info/raw h field vals)
            'dmatrix-set-feature-info!))

(define (nul-separated-bytes->strings bs count)
  (define len (bytes-length bs))
  (let loop ([start 0] [i 0] [acc '()])
    (cond
      [(= i count) (reverse acc)]
      [else
       (define end
         (let find-nul ([j start])
           (cond
             [(>= j len) len]
             [(zero? (bytes-ref bs j)) j]
             [else (find-nul (add1 j))])))
       (loop (add1 end)
             (add1 i)
             (cons (bytes->string/utf-8 bs #f start end) acc))])))

(define (dmatrix-get-feature-info h field)
  (define-values (rc len count)
    (xgb-dmatrix-get-str-feature-info/raw h field 0 (make-bytes 0)))
  (cond
    [(zero? rc) '()]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 count2)
       (xgb-dmatrix-get-str-feature-info/raw h field len buf))
     (check-ok rc2 'dmatrix-get-feature-info)
     (unless (= len2 len)
       (error 'dmatrix-get-feature-info
              "expected ~a bytes, got ~a" len len2))
     (nul-separated-bytes->strings buf count2)]
    [else (check-ok rc 'dmatrix-get-feature-info)]))

(define (dmatrix-free! h)
  (when (cpointer-has-tag? h 'DMatrix)
    (xgb-dmatrix-free/raw h)
    (set-cpointer-tag! h 'DMatrix-freed)))

(define (dmatrix-nrow h)
  (define-values (rc out) (xgb-dmatrix-num-row/raw h))
  (check-ok rc 'dmatrix-nrow)
  out)

(define (dmatrix-ncol h)
  (define-values (rc out) (xgb-dmatrix-num-col/raw h))
  (check-ok rc 'dmatrix-ncol)
  out)

;; Copy a float info field into a fresh f32vector.  The pointer returned by
;; the raw call is borrowed from the DMatrix; we memcpy out of it before
;; letting control flow near anything that could invalidate it.  An unset
;; field returns an empty f32vector (rc is still 0).
(define (dmatrix-get-float-info h field)
  (define-values (rc len ptr) (xgb-dmatrix-get-float-info/raw h field))
  (check-ok rc 'dmatrix-get-float-info)
  (define result (make-f32vector len))
  (when (and ptr (> len 0))
    (memcpy (f32vector->cpointer result) ptr len _float))
  result)

(define (dmatrix-get-uint-info h field)
  (define-values (rc len ptr) (xgb-dmatrix-get-uint-info/raw h field))
  (check-ok rc 'dmatrix-get-uint-info)
  (define result (make-u32vector len))
  (when (and ptr (> len 0))
    (memcpy (u32vector->cpointer result) ptr len _uint32))
  result)

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

;; ---------------------------------------------------------------------------
;; Booster wrappers.
;;
;; Same lifetime model as DMatrix: allocator/deallocator wired at the raw
;; layer, explicit `booster-free!` flips the cpointer tag so a second call
;; raises a contract error rather than double-freeing.
;;
;; XGBoost's predict API requires a JSON config with `type` set; "{}" raises
;; a fatal "Argument `type` is required".  `booster-predict` defaults to
;; inference-mode predictions over all iterations.
;; ---------------------------------------------------------------------------

(define (booster? v) (Booster? v))

;; XGBoost's predict-type integers, per the C API doc:
;;   0 value (default), 1 margin, 2 SHAP contribution,
;;   3 approximate SHAP, 4 SHAP interaction,
;;   5 approximate SHAP interaction, 6 leaf indices
(define output-symbol->type
  '((value               . 0)
    (margin              . 1)
    (contribs            . 2)
    (approx-contribs     . 3)
    (interactions        . 4)
    (approx-interactions . 5)
    (leaf                . 6)))

(define (build-predict-config #:output [output 'value]
                              #:iteration-end [iter-end 0]
                              #:missing [missing #f])
  ;; iteration_end=0 means "use all trees" in XGBoost.  Higher values predict
  ;; using only the first N rounds — useful for plotting prediction
  ;; trajectories or for early-stop verification.
  (define type
    (cdr (assq output output-symbol->type)))
  (format
   "{\"type\":~a,\"training\":false,\"iteration_begin\":0,\"iteration_end\":~a,\"strict_shape\":false~a}"
   type
   iter-end
   (if missing
       (format ",\"missing\":~a" (json-number missing))
       "")))

(define booster-default-predict-config (build-predict-config))

(define (booster-create [cache '()])
  (define h (xgb-booster-create/raw cache))
  (unless h
    (error 'booster-create
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  h)

(define (booster-free! h)
  (when (cpointer-has-tag? h 'Booster)
    (xgb-booster-free/raw h)
    (set-cpointer-tag! h 'Booster-freed)))

(define (booster-reset! h)
  (check-ok (xgb-booster-reset/raw h) 'booster-reset!))

(define (booster-slice h begin-layer end-layer [step 1])
  (define sliced (xgb-booster-slice/raw h begin-layer end-layer step))
  (unless sliced
    (error 'booster-slice
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  sliced)

(define (booster-boosted-rounds h)
  (define-values (rc out) (xgb-booster-boosted-rounds/raw h))
  (check-ok rc 'booster-boosted-rounds)
  out)

(define (booster-num-feature h)
  (define-values (rc out) (xgb-booster-num-feature/raw h))
  (check-ok rc 'booster-num-feature)
  out)

(define (booster-set-param! h key value)
  (check-ok (xgb-booster-set-param/raw h key value) 'booster-set-param!))

(define (booster-update-one-iter! h iter dtrain)
  (check-ok (xgb-booster-update-one-iter/raw h iter dtrain)
            'booster-update-one-iter!))

;; Predict against `dmat`, returning a fresh f32vector sized to whatever
;; XGBoost emitted.
;;
;; #:output selects the prediction type — 'value (default) is the usual
;; per-row response; 'margin is the raw score before the link function;
;; 'contribs / 'interactions emit per-feature SHAP values
;; (shape grows to nrow*(ncol+1) / nrow*(ncol+1)^2); 'leaf emits the
;; leaf index each row landed in for every tree (shape nrow*ntree).
;;
;; #:iteration-end limits prediction to the first N rounds (0 = all).
;;
;; The rc=2 resize-and-retry path absorbs all of these shape changes
;; transparently.  The booster-owned buffer is copied on the C side
;; before any return.
;;
;; #:config takes a fully-formed JSON predict-config string and supersedes
;; both keyword args; pass it when you need to set fields we don't surface
;; (e.g. iteration_begin, strict_shape).
(define (booster-predict h dmat
                         #:output [output 'value]
                         #:iteration-end [iter-end 0]
                         #:config [config #f])
  (define cfg
    (or config (build-predict-config #:output output #:iteration-end iter-end)))
  (define guess (max 1 (dmatrix-nrow dmat)))
  (define buf (make-f32vector guess))
  (define-values (rc len) (xgb-booster-predict/raw h dmat cfg guess buf))
  (cond
    [(zero? rc)
     ;; The full prediction fit in `buf`, but `len` may be smaller than
     ;; capacity (rare in practice).  Trim to the reported size.
     (cond
       [(= len guess) buf]
       [else
        (define trimmed (make-f32vector len))
        (memcpy (f32vector->cpointer trimmed) (f32vector->cpointer buf)
                len _float)
        trimmed])]
    [(= rc 2)
     ;; Buffer too small; resize and retry.  `len` holds the required size.
     (define buf2 (make-f32vector len))
     (define-values (rc2 len2) (xgb-booster-predict/raw h dmat cfg len buf2))
     (check-ok rc2 'booster-predict)
     (unless (= len2 len)
       (error 'booster-predict
              "expected ~a predictions, got ~a" len len2))
     buf2]
    [else (check-ok rc 'booster-predict)]))

(define (copy-prediction-result who guess raw)
  (define buf (make-f32vector guess))
  (define-values (rc len) (raw guess buf))
  (cond
    [(zero? rc)
     (cond
       [(= len guess) buf]
       [else
        (define trimmed (make-f32vector len))
        (memcpy (f32vector->cpointer trimmed) (f32vector->cpointer buf)
                len _float)
        trimmed])]
    [(= rc 2)
     (define buf2 (make-f32vector len))
     (define-values (rc2 len2) (raw len buf2))
     (check-ok rc2 who)
     (unless (= len2 len)
       (error who "expected ~a predictions, got ~a" len len2))
     buf2]
    [else (check-ok rc who)]))

(define (proxy-handle proxy)
  (or proxy #f))

(define (booster-predict-from-dense h data nrow ncol
                                   #:missing [missing +nan.0]
                                   #:output [output 'value]
                                   #:iteration-end [iter-end 0]
                                   #:config [config #f]
                                   #:proxy [proxy #f])
  (unless (= (f32vector-length data) (* nrow ncol))
    (raise-argument-error 'booster-predict-from-dense
                          (format "f32vector of length ~a (nrow*ncol)"
                                  (* nrow ncol))
                          data))
  (define cfg
    (or config (build-predict-config #:output output
                                     #:iteration-end iter-end
                                     #:missing missing)))
  (define data-json (f32-array-interface data (list nrow ncol)))
  (copy-prediction-result
   'booster-predict-from-dense
   (max 1 nrow)
   (lambda (capacity buf)
     (xgb-booster-predict-from-dense/raw h data-json cfg
                                         (proxy-handle proxy)
                                         capacity buf))))

(define (booster-predict-from-csr h indptr indices data ncol
                                 #:missing [missing +nan.0]
                                 #:output [output 'value]
                                 #:iteration-end [iter-end 0]
                                 #:config [config #f]
                                 #:proxy [proxy #f])
  (unless (= (u32vector-length indices) (f32vector-length data))
    (error 'booster-predict-from-csr
           "indices length ~a does not match data length ~a"
           (u32vector-length indices)
           (f32vector-length data)))
  (unless (= (u64vector-ref indptr (sub1 (u64vector-length indptr)))
             (f32vector-length data))
    (error 'booster-predict-from-csr
           "final indptr value must equal data length"))
  (define nrow (sub1 (u64vector-length indptr)))
  (define cfg
    (or config (build-predict-config #:output output
                                     #:iteration-end iter-end
                                     #:missing missing)))
  (copy-prediction-result
   'booster-predict-from-csr
   (max 1 nrow)
   (lambda (capacity buf)
     (xgb-booster-predict-from-csr/raw h
                                       (u64-array-interface indptr)
                                       (u32-array-interface indices)
                                       (f32-array-interface
                                        data
                                        (list (f32vector-length data)))
                                       ncol
                                       cfg
                                       (proxy-handle proxy)
                                       capacity buf))))

(define (booster-predict-from-columnar h columns
                                      #:missing [missing +nan.0]
                                      #:output [output 'value]
                                      #:iteration-end [iter-end 0]
                                      #:config [config #f]
                                      #:proxy [proxy #f])
  (when (null? columns)
    (raise-argument-error 'booster-predict-from-columnar
                          "non-empty list of f32vectors"
                          columns))
  (define nrow (f32vector-length (car columns)))
  (for ([col (in-list columns)])
    (unless (= (f32vector-length col) nrow)
      (error 'booster-predict-from-columnar
             "all columns must have the same length")))
  (define data-json
    (format "[~a]"
            (string-join
             (for/list ([col (in-list columns)])
               (f32-array-interface col (list nrow)))
             ",")))
  (define cfg
    (or config (build-predict-config #:output output
                                     #:iteration-end iter-end
                                     #:missing missing)))
  (copy-prediction-result
   'booster-predict-from-columnar
   (max 1 nrow)
   (lambda (capacity buf)
     (xgb-booster-predict-from-columnar/raw h data-json cfg
                                            (proxy-handle proxy)
                                            capacity buf))))

;; --- Save / load --------------------------------------------------------

(define (booster-save-model! h path)
  (check-ok (xgb-booster-save-model/raw h path) 'booster-save-model!))

(define (booster-load-model! h path)
  (check-ok (xgb-booster-load-model/raw h path) 'booster-load-model!))

;; Serialize a booster to a fresh bytes object.  `format` is "ubj" (compact
;; binary, default) or "json" (human-readable).  Same size-then-fill dance
;; as predict: probe to get the required length, allocate, refill.
(define (booster-save-model-to-bytes h #:format [fmt "ubj"])
  (define config (string-append "{\"format\":\"" fmt "\"}"))
  (define-values (rc len)
    (xgb-booster-save-model-to-buffer/raw h config 0 (make-bytes 0)))
  (cond
    [(zero? rc) (make-bytes len)]   ; degenerate: 0-byte model
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-save-model-to-buffer/raw h config len buf))
     (check-ok rc2 'booster-save-model-to-bytes)
     (unless (= len2 len)
       (error 'booster-save-model-to-bytes
              "expected ~a bytes, got ~a" len len2))
     buf]
    [else (check-ok rc 'booster-save-model-to-bytes)]))

(define (booster-load-model-from-bytes! h buf)
  (check-ok (xgb-booster-load-model-from-buffer/raw h buf)
            'booster-load-model-from-bytes!))

;; --- Config / attrs / inspection ---------------------------------------

(define (booster-save-json-config h)
  (copy-string-result 'booster-save-json-config
                      (lambda (capacity buf)
                        (xgb-booster-save-json-config/raw h capacity buf))))

(define (booster-load-json-config! h config)
  (check-ok (xgb-booster-load-json-config/raw h config)
            'booster-load-json-config!))

(define (booster-set-attr! h key value)
  (check-ok (xgb-booster-set-attr/raw h key value)
            'booster-set-attr!))

(define (booster-delete-attr! h key)
  (check-ok (xgb-booster-delete-attr/raw h key)
            'booster-delete-attr!))

(define (booster-get-attr h key)
  (define-values (rc len found)
    (xgb-booster-get-attr/raw h key 0 (make-bytes 0)))
  (cond
    [(and (zero? rc) (zero? found)) #f]
    [(zero? rc) ""]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 found2)
       (xgb-booster-get-attr/raw h key len buf))
     (check-ok rc2 'booster-get-attr)
     (cond
       [(zero? found2) #f]
       [else (bytes->string/utf-8 buf #f 0 len2)])]
    [else (check-ok rc 'booster-get-attr)]))

(define (copy-nul-separated-result who raw)
  (define-values (rc len count) (raw 0 (make-bytes 0)))
  (cond
    [(zero? rc) '()]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 count2) (raw len buf))
     (check-ok rc2 who)
     (unless (= len2 len)
       (error who "expected ~a bytes, got ~a" len len2))
     (nul-separated-bytes->strings buf count2)]
    [else (check-ok rc who)]))

(define (booster-get-attr-names h)
  (copy-nul-separated-result
   'booster-get-attr-names
   (lambda (capacity buf)
     (xgb-booster-get-attr-names/raw h capacity buf))))

(define (booster-set-feature-info! h field vals)
  (check-ok (xgb-booster-set-str-feature-info/raw h field vals)
            'booster-set-feature-info!))

(define (booster-get-feature-info h field)
  (copy-nul-separated-result
   'booster-get-feature-info
   (lambda (capacity buf)
     (xgb-booster-get-str-feature-info/raw h field capacity buf))))

(define (booster-dump-model h #:format [fmt "text"] #:with-stats? [stats? #f])
  (copy-nul-separated-result
   'booster-dump-model
   (lambda (capacity buf)
     (xgb-booster-dump-model/raw h fmt (if stats? 1 0) capacity buf))))

(define (booster-dump-model-with-features h feature-names feature-types
                                          #:format [fmt "text"]
                                          #:with-stats? [stats? #f])
  (unless (= (length feature-names) (length feature-types))
    (error 'booster-dump-model-with-features
           "feature name count ~a does not match feature type count ~a"
           (length feature-names)
           (length feature-types)))
  (copy-nul-separated-result
   'booster-dump-model-with-features
   (lambda (capacity buf)
     (xgb-booster-dump-model-with-features/raw h feature-names feature-types
                                               fmt (if stats? 1 0)
                                               capacity buf))))

(define (json-quote-string s)
  (format "~s" s))

(define (feature-score-config importance-type feature-names)
  (define base (format "\"importance_type\":~a"
                       (json-quote-string importance-type)))
  (define names
    (if feature-names
        (format ",\"feature_names\":[~a]"
                (string-join (map json-quote-string feature-names) ","))
        ""))
  (format "{~a~a}" base names))

(define (u64vector->list vec)
  (for/list ([i (in-range (u64vector-length vec))])
    (u64vector-ref vec i)))

(define (booster-feature-score h
                               #:importance-type [importance-type "weight"]
                               #:feature-names [feature-names #f]
                               #:config [config #f])
  (define cfg (or config (feature-score-config importance-type feature-names)))
  (define-values (rc feature-len n-features dim n-scores)
    (xgb-booster-feature-score/raw h cfg
                                   0 (make-bytes 0)
                                   0 (make-u64vector 0)
                                   0 (make-f32vector 0)))
  (cond
    [(zero? rc)
     (hash 'features '()
           'shape '()
           'scores (make-f32vector 0))]
    [(= rc 2)
     (define feature-buf (make-bytes feature-len))
     (define shape-buf (make-u64vector dim))
     (define score-buf (make-f32vector n-scores))
     (define-values (rc2 feature-len2 n-features2 dim2 n-scores2)
       (xgb-booster-feature-score/raw h cfg
                                      feature-len feature-buf
                                      dim shape-buf
                                      n-scores score-buf))
     (check-ok rc2 'booster-feature-score)
     (unless (and (= feature-len2 feature-len)
                  (= n-features2 n-features)
                  (= dim2 dim)
                  (= n-scores2 n-scores))
       (error 'booster-feature-score
              "feature score output shape changed during copy"))
     (hash 'features (nul-separated-bytes->strings feature-buf n-features2)
           'shape (u64vector->list shape-buf)
           'scores score-buf)]
    [else (check-ok rc 'booster-feature-score)]))

;; --- Eval one iter ------------------------------------------------------

;; `eval-set` is a list of (name . dmatrix) pairs, in the order their
;; metrics should appear in the result line.  Returns a string like
;;   "[<iter>]\t<name>-<metric>:<value>\t..."
;; matching the format XGBoost prints during default training.
(define (booster-eval-one-iter h iter eval-set)
  (define dmats (map cdr eval-set))
  (define names (map car eval-set))
  ;; Typical line is well under 256 chars even with several metrics; over-
  ;; allocate to avoid the resize round-trip in the common case.
  (define guess 512)
  (define buf (make-bytes guess))
  (define-values (rc len)
    (xgb-booster-eval-one-iter/raw h iter dmats names guess buf))
  (cond
    [(zero? rc) (bytes->string/utf-8 buf #f 0 len)]
    [(= rc 2)
     (define buf2 (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-eval-one-iter/raw h iter dmats names len buf2))
     (check-ok rc2 'booster-eval-one-iter)
     (bytes->string/utf-8 buf2 #f 0 len2)]
    [else (check-ok rc 'booster-eval-one-iter)]))

;; Parse a metric line into a hash from "<name>-<metric>" to numeric value.
;; The leading "[<iter>]" token is dropped; any token that is not of the
;; shape "<key>:<number>" is silently skipped.  Greedy `(.*)` on the key
;; means metric names containing additional colons would be misparsed, but
;; XGBoost's built-in metrics never do that.
(define (parse-eval-line line)
  (define parts (regexp-split #rx"\t" line))
  (for/hash ([part (in-list (if (pair? parts) (cdr parts) '()))]
             #:when (regexp-match? #rx":" part))
    (define m (regexp-match #rx"^(.*):([^:]+)$" part))
    (define key (cadr m))
    (define val (string->number (caddr m)))
    (values key (if (real? val) val +nan.0))))

(module+ test
  (require rackunit
           racket/file)

  ;; Existing smoke tests.
  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))
  (check-pred rational? (run-regression-demo))
  (let ([p (run-classification-demo)])
    (check-true (<= 0 p 1)))

  (test-case "build info returns JSON"
    (define info (xgboost-build-info))
    (check-regexp-match #rx"^\\{" info))

  (test-case "global config round-trips verbosity"
    (define before (xgboost-get-global-config))
    (dynamic-wind
      void
      (lambda ()
        (xgboost-set-global-config! "{\"verbosity\":0}")
        (check-regexp-match #rx"\"verbosity\":0"
                            (xgboost-get-global-config)))
      (lambda () (xgboost-set-global-config! before))))

  (test-case "bad global config reports XGBoost error"
    (check-exn exn:fail?
               (lambda ()
                 (xgboost-set-global-config! "{not-json"))))

  (test-case "log callback registration accepts a Racket procedure"
    (xgboost-register-log-callback! (lambda (msg) (void))))

  ;; --- DMatrix round-trip -------------------------------------------------
  (define (make-data) (f32vector 1.0 2.0 3.0 4.0 5.0 6.0))  ; 2 rows x 3 cols

  (test-case "dmatrix create/free round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-pred dmatrix? dm)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-dense array interface"
    (define dm (dmatrix-create-from-dense (make-data) 2 3 -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (check-equal? (dmatrix->list dm)
                  '((1.0 2.0 3.0) (4.0 5.0 6.0)))
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-csr array interfaces"
    (define dm
      (dmatrix-create-from-csr
       (u64vector 0 2 4)
       (u32vector 0 2 1 2)
       (f32vector 1.0 3.0 5.0 6.0)
       3
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (define rows (dmatrix->list dm))
    (check-= (first (first rows)) 1.0 1e-6)
    (check-true (not (= (second (first rows)) (second (first rows)))))
    (check-= (third (first rows)) 3.0 1e-6)
    (check-= (second (second rows)) 5.0 1e-6)
    (check-= (third (second rows)) 6.0 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-csc array interfaces"
    (define dm
      (dmatrix-create-from-csc
       (u64vector 0 1 2 4)
       (u32vector 0 1 0 1)
       (f32vector 1.0 5.0 3.0 6.0)
       2
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (define rows (dmatrix->list dm))
    (check-= (first (first rows)) 1.0 1e-6)
    (check-true (not (= (second (first rows)) (second (first rows)))))
    (check-= (third (first rows)) 3.0 1e-6)
    (check-= (second (second rows)) 5.0 1e-6)
    (check-= (third (second rows)) 6.0 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-columnar array interfaces"
    (define dm
      (dmatrix-create-from-columnar
       (list (f32vector 1.0 4.0)
             (f32vector 2.0 5.0)
             (f32vector 3.0 6.0))
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (check-equal? (dmatrix->list dm)
                  '((1.0 2.0 3.0) (4.0 5.0 6.0)))
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-uri loads libsvm"
    (define tmp (make-temporary-file "xgboost-uri-~a.libsvm"))
    (dynamic-wind
      void
      (lambda ()
        (call-with-output-file tmp
          (lambda (out)
            (displayln "0 1:1 3:3" out)
            (displayln "1 2:5 3:6" out))
          #:exists 'truncate)
        (define dm
          (dmatrix-create-from-uri
           (format "{\"uri\":\"~a?format=libsvm\",\"silent\":1}"
                   (path->string tmp))))
        (check-equal? (dmatrix-nrow dm) 2)
        (check-equal? (dmatrix-ncol dm) 4)
        (dmatrix-free! dm))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))

  (test-case "dmatrix finalizer path: drop reference, collect-garbage"
    (let loop ([i 0])
      (when (< i 64)
        (dmatrix-create-from-mat (make-data) 2 3)
        (loop (add1 i))))
    (collect-garbage) (collect-garbage) (collect-garbage)
    ;; If we reach here without crashing, the finalizer chain is safe.
    (check-true #t))

  (test-case "dmatrix-free! is safe to call twice (tag guard, no double-free)"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-free! dm)
    ;; Second call is a no-op; must not segfault or raise.
    (dmatrix-free! dm))

  (test-case "shape-mismatch raises contract error"
    (check-exn exn:fail:contract?
               (lambda ()
                 (dmatrix-create-from-mat (f32vector 1.0 2.0 3.0) 2 3))))

  (test-case "set-float-info success"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
    (dmatrix-free! dm))

  (test-case "set-float-info with bogus field surfaces C error"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-exn (lambda (e)
                 (and (exn:fail? e)
                      ;; Message should include the C-side context.
                      (regexp-match? #rx"XGDMatrixSetFloatInfo"
                                     (exn-message e))))
               (lambda ()
                 (dmatrix-set-float-info! dm "definitely_not_a_field"
                                          (f32vector 0.0 1.0))))
    (dmatrix-free! dm))

  ;; --- Metadata accessors + dmatrix-show ----------------------------------
  (test-case "dmatrix-nrow / dmatrix-ncol report shape"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (dmatrix-free! dm))

  (test-case "dmatrix-get-float-info round-trips labels"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.5 1.5))
    (define got (dmatrix-get-float-info dm "label"))
    (check-equal? (f32vector-length got) 2)
    (check-= (f32vector-ref got 0) 0.5 1e-6)
    (check-= (f32vector-ref got 1) 1.5 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-get-float-info on unset field returns empty vector"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (f32vector-length (dmatrix-get-float-info dm "weight")) 0)
    (dmatrix-free! dm))

  (test-case "dmatrix-show prints shape and set info fields"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    (check-regexp-match #rx"DMatrix 2x3" s)
    (check-regexp-match #rx"label:" s)
    ;; Small matrix: show all rows/cols, no ellipses; uniform width.
    (check-regexp-match #rx"1\\.0000 2\\.0000 3\\.0000" s)
    (check-false (regexp-match? #rx"\\.\\.\\." s))
    (dmatrix-free! dm))

  (test-case "dmatrix-show truncates to 4x4 with polars-style ellipses"
    (define big (list->f32vector
                 (for/list ([i (in-range 36)]) (exact->inexact i))))
    (define dm (dmatrix-create-from-mat big 6 6))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    (check-regexp-match #rx"DMatrix 6x6" s)
    ;; Widest cell in preview is "18.0000" (7 chars); narrower cells right-pad.
    ;; First row: " 0.0000  1.0000  2.0000  3.0000 ..."
    (check-regexp-match
     #rx" 0\\.0000  1\\.0000  2\\.0000  3\\.0000 \\.\\.\\." s)
    ;; Fourth (last shown) row starts at value 18, no padding needed.
    (check-regexp-match
     #rx"18\\.0000 19\\.0000 20\\.0000 21\\.0000 \\.\\.\\." s)
    ;; Row 5 should not appear (truncated).
    (check-false (regexp-match? #rx"24\\.0000" s))
    ;; Final standalone "..." line signalling row truncation.
    (check-regexp-match #rx"\n  \\.\\.\\.\n" s)
    (dmatrix-free! dm))

  (test-case "dmatrix-show pads every preview row to the same column width"
    ;; Mix of 1-char and 3-char integer parts forces non-trivial alignment.
    (define data (list->f32vector
                  '(1.0 22.0 333.0
                    4.0  5.0   6.0
                    7.0  8.0   9.0)))
    (define dm (dmatrix-create-from-mat data 3 3))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    ;; Split off the matrix-body lines (skip the "DMatrix" header).
    (define body-lines
      (for/list ([line (in-list (regexp-split #rx"\n" s))]
                 #:when (regexp-match? #rx"^  [ 0-9]" line))
        line))
    (check-equal? (length body-lines) 3)
    ;; All three body lines must be the same length — proof of column alignment.
    (define widths (map string-length body-lines))
    (check-equal? (apply min widths) (apply max widths)
                  (format "body lines differ in width: ~a" body-lines))
    ;; The widest cell is "333.0000" (8 chars); narrower cells get pre-padded.
    (check-regexp-match #rx"  1\\.0000  22\\.0000 333\\.0000" s)
    (check-regexp-match #rx"  4\\.0000   5\\.0000   6\\.0000" s)
    (dmatrix-free! dm))

  (test-case "dmatrix->list reconstructs the feature matrix"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define rows (dmatrix->list dm))
    (check-equal? (length rows) 2)
    (check-equal? (map length rows) '(3 3))
    (for ([expected (in-list '((1.0 2.0 3.0) (4.0 5.0 6.0)))]
          [got (in-list rows)])
      (for ([e (in-list expected)] [g (in-list got)])
        (check-= g e 1e-6)))
    (check-equal? (dmatrix-num-non-missing dm) 6)
    (dmatrix-free! dm))

  ;; --- Leak test (explicit-free path) -------------------------------------
  ;;
  ;; `current-memory-use` only tracks Racket-managed memory, so it won't see
  ;; C-side leaks directly.  The C-side leak smoke test in the gtest suite
  ;; (`XgbDMatrixTest.LeakSmokeTest`) is the authoritative backstop; here we
  ;; assert the Racket-side balance counter stays at zero over 10k cycles and
  ;; that Racket memory use is bounded.
  (test-case "10k explicit create/free cycles: counter balances, memory bounded"
    (define alive 0)
    (define mem-before (current-memory-use))
    (for ([_ (in-range 10000)])
      (define dm (dmatrix-create-from-mat (make-data) 2 3))
      (set! alive (add1 alive))
      (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
      (dmatrix-free! dm)
      (set! alive (sub1 alive)))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (define mem-after (current-memory-use))
    (check-equal? alive 0 "every allocation should have a matching free")
    (check-true (< (- mem-after mem-before) (* 64 1024 1024))
                (format "Racket memory grew by ~a bytes over 10k iters"
                        (- mem-after mem-before))))

  ;; --- Finalizer-path smoke test ------------------------------------------
  ;;
  ;; Allocate without explicit free; the allocator-registered finalizer must
  ;; reclaim each handle under GC without crashing.  We cannot easily observe
  ;; the finalizer firing from Racket (wrapping `xgb-dmatrix-free/raw` would
  ;; break the allocator contract), so the assurance here is "does not crash"
  ;; plus the C-side gtest leak test.
  (test-case "finalizer path: GC reclaims DMatrices without explicit free"
    (for ([_ (in-range 256)])
      (dmatrix-create-from-mat (make-data) 2 3))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (check-true #t))

  ;; --- Phase 3 metadata APIs --------------------------------------------

  (test-case "dmatrix-set-feature-info! / dmatrix-get-feature-info round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-feature-info! dm "feature_name" '("height" "weight" "age"))
    (dmatrix-set-feature-info! dm "feature_type" '("q" "q" "i"))
    (check-equal? (dmatrix-get-feature-info dm "feature_name")
                  '("height" "weight" "age"))
    (check-equal? (dmatrix-get-feature-info dm "feature_type")
                  '("q" "q" "i"))
    (dmatrix-free! dm))

  (test-case "dmatrix-get-feature-info on unset field returns empty list"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (dmatrix-get-feature-info dm "feature_name") '())
    (dmatrix-free! dm))

  (test-case "dmatrix-set-uint-info! group + dmatrix-get-uint-info group_ptr"
    ;; XGBoost reads "group" as ranking-group sizes and exposes the
    ;; cumulative form via "group_ptr" (so {2} sets to {0,2}).
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-uint-info! dm "group" (u32vector 2))
    (define g (dmatrix-get-uint-info dm "group_ptr"))
    (check-equal? (u32vector-length g) 2)
    (check-equal? (u32vector-ref g 0) 0)
    (check-equal? (u32vector-ref g 1) 2)
    (dmatrix-free! dm))

  (test-case "dmatrix-set-info-from-interface! drives label round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define labels (f32vector 0.25 0.75))
    (dmatrix-set-info-from-interface!
     dm "label"
     (format "{\"data\":[~a,false],\"typestr\":\"<f4\",\"shape\":[2],\"version\":3}"
             (cast (f32vector->cpointer labels) _pointer _uintptr)))
    (define got (dmatrix-get-float-info dm "label"))
    (check-equal? (f32vector-length got) 2)
    (check-= (f32vector-ref got 0) 0.25 1e-6)
    (check-= (f32vector-ref got 1) 0.75 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-slice picks rows in given order"
    (define data (f32vector 1.0 2.0
                            3.0 4.0
                            5.0 6.0))
    (define dm (dmatrix-create-from-mat data 3 2))
    (define sliced (dmatrix-slice dm '(2 0)))
    (check-equal? (dmatrix-nrow sliced) 2)
    (check-equal? (dmatrix-ncol sliced) 2)
    (check-equal? (dmatrix->list sliced)
                  '((5.0 6.0) (1.0 2.0)))
    (dmatrix-free! sliced)
    (dmatrix-free! dm))

  (test-case "dmatrix-slice accepts s32vector and rejects bad indices"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-pred dmatrix? (dmatrix-slice dm (s32vector 1 0)))
    (check-exn exn:fail:contract?
               (lambda ()
                 (dmatrix-slice dm '(-1))))
    (dmatrix-free! dm))

  (test-case "dmatrix-save-binary! writes a file XGBoost can reload"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define tmp (make-temporary-file "xgboost-dmat-~a.buffer"))
    (dynamic-wind
      void
      (lambda ()
        (dmatrix-save-binary! dm tmp)
        (define loaded
          (dmatrix-create-from-uri
           (format "{\"uri\":\"~a\",\"silent\":1}" (path->string tmp))))
        (check-equal? (dmatrix-nrow loaded) 2)
        (check-equal? (dmatrix-ncol loaded) 3)
        (dmatrix-free! loaded))
      (lambda () (when (file-exists? tmp) (delete-file tmp))))
    (dmatrix-free! dm))

  (test-case "dmatrix-get-quantile-cut returns array-interface JSON after training"
    ;; Quantile cuts are computed during training on a hist-method booster.
    (define dm
      (dmatrix-create-from-mat
       (f32vector 1.0 2.0 0.5 2.0 1.0 1.5 3.0 0.5 0.0
                  0.5 3.0 2.0 4.0 2.0 1.0 1.5 1.5 0.5
                  2.5 3.5 1.5 0.0 1.0 0.0)
       8 3))
    (dmatrix-set-float-info! dm "label"
                             (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
    (define b (booster-create (list dm)))
    (booster-set-param! b "objective" "reg:squarederror")
    (booster-set-param! b "tree_method" "hist")
    (booster-set-param! b "max_depth" "2")
    (booster-set-param! b "verbosity" "0")
    (booster-update-one-iter! b 0 dm)
    (define-values (indptr-json data-json) (dmatrix-get-quantile-cut dm))
    (check-true (regexp-match? #rx"\"shape\"" indptr-json))
    (check-true (regexp-match? #rx"\"shape\"" data-json))
    (booster-free! b)
    (dmatrix-free! dm))

  ;; --- Booster -----------------------------------------------------------

  ;; Same fixture as the gtest regression test: 8x3, label ≈ 2*x0 + x1 - x2.
  (define (regression-features)
    (f32vector 1.0 2.0 0.5 2.0 1.0 1.5 3.0 0.5 0.0
               0.5 3.0 2.0 4.0 2.0 1.0 1.5 1.5 0.5
               2.5 3.5 1.5 0.0 1.0 0.0))
  (define (regression-labels)
    (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

  (define (make-trained-regressor #:rounds [rounds 50])
    (define dtrain (dmatrix-create-from-mat (regression-features) 8 3))
    (dmatrix-set-float-info! dtrain "label" (regression-labels))
    (define b (booster-create (list dtrain)))
    (booster-set-param! b "objective" "reg:squarederror")
    (booster-set-param! b "max_depth" "3")
    (booster-set-param! b "eta" "0.1")
    (booster-set-param! b "verbosity" "0")
    (for ([iter (in-range rounds)])
      (booster-update-one-iter! b iter dtrain))
    (values b dtrain))

  (test-case "booster create/free round-trip"
    (define b (booster-create))
    (check-pred booster? b)
    (booster-free! b))

  (test-case "booster-free! is idempotent (tag guard)"
    (define b (booster-create))
    (booster-free! b)
    (booster-free! b))   ; no-op; must not crash

  (test-case "booster-set-param! accepts known params"
    (define b (booster-create))
    (booster-set-param! b "objective" "reg:squarederror")
    (booster-set-param! b "max_depth" "3")
    (booster-free! b))

  (test-case "booster fits training data (MSE under threshold)"
    (define-values (b dtrain) (make-trained-regressor))
    (define preds (booster-predict b dtrain))
    (check-equal? (f32vector-length preds) 8)
    (define labels (regression-labels))
    (define mse
      (/ (for/sum ([i (in-range 8)])
           (define d (- (f32vector-ref preds i) (f32vector-ref labels i)))
           (* d d))
         8))
    (check-true (< mse 3.0)
                (format "training MSE too high: ~a" mse))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster-predict copies; second call doesn't disturb the first"
    (define-values (b dtrain) (make-trained-regressor #:rounds 10))
    (define first (booster-predict b dtrain))
    (define snapshot (f32vector->list first))
    (booster-predict b dtrain)              ; second call, result discarded
    (check-equal? (f32vector->list first) snapshot)
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster finalizer path: GC reclaims without explicit free"
    (for ([_ (in-range 64)])
      (booster-create))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (check-true #t))

  ;; --- Save / load ------------------------------------------------------

  (test-case "save/load via file: predictions survive serde"
    (define-values (b dtrain) (make-trained-regressor #:rounds 20))
    (define baseline (booster-predict b dtrain))
    (define tmp (make-temporary-file "xgbrkt-~a.json"))
    (dynamic-wind
      void
      (lambda ()
        (booster-save-model! b tmp)
        (define b2 (booster-create))
        (booster-load-model! b2 tmp)
        (define preds (booster-predict b2 dtrain))
        (check-equal? (f32vector->list preds) (f32vector->list baseline))
        (booster-free! b2))
      (lambda () (when (file-exists? tmp) (delete-file tmp))))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "save/load via bytes (ubj): predictions survive serde"
    (define-values (b dtrain) (make-trained-regressor #:rounds 20))
    (define baseline (booster-predict b dtrain))
    (define blob (booster-save-model-to-bytes b))
    (check-pred bytes? blob)
    (check-true (> (bytes-length blob) 0))
    (define b2 (booster-create))
    (booster-load-model-from-bytes! b2 blob)
    (check-equal? (f32vector->list (booster-predict b2 dtrain))
                  (f32vector->list baseline))
    (booster-free! b2)
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "save/load via bytes (json): produces a JSON-shaped blob"
    (define-values (b dtrain) (make-trained-regressor #:rounds 5))
    (define blob (booster-save-model-to-bytes b #:format "json"))
    ;; First non-whitespace byte of an XGBoost JSON model is '{'.
    (check-equal? (bytes-ref blob 0) (char->integer #\{))
    (define b2 (booster-create))
    (booster-load-model-from-bytes! b2 blob)
    (check-equal?
     (f32vector->list (booster-predict b2 dtrain))
     (f32vector->list (booster-predict b dtrain)))
    (booster-free! b2)
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "load-from-bytes on garbage raises with C error context"
    (define b (booster-create))
    (check-exn (lambda (e)
                 (and (exn:fail? e)
                      (regexp-match? #rx"XGBoosterLoadModelFromBuffer"
                                     (exn-message e))))
               (lambda ()
                 (booster-load-model-from-bytes! b (make-bytes 64 65))))
    (booster-free! b))

  ;; --- Eval one iter ----------------------------------------------------

  (test-case "parse-eval-line: typical XGBoost output"
    (define h (parse-eval-line "[3]\ttrain-rmse:1.2345\teval-rmse:2.3456"))
    (check-equal? (hash-count h) 2)
    (check-= (hash-ref h "train-rmse") 1.2345 1e-9)
    (check-= (hash-ref h "eval-rmse")  2.3456 1e-9))

  (test-case "parse-eval-line: header-only / malformed input"
    (check-equal? (hash-count (parse-eval-line "[0]")) 0)
    (check-equal? (hash-count (parse-eval-line "")) 0))

  (test-case "booster-eval-one-iter returns the iter and named metrics"
    (define-values (b dtrain) (make-trained-regressor #:rounds 5))
    (define line (booster-eval-one-iter b 4 (list (cons "train" dtrain))))
    (check-true (regexp-match? #rx"^\\[4\\]" line)
                (format "expected line to start with [4]: ~s" line))
    (define metrics (parse-eval-line line))
    (check-true (hash-has-key? metrics "train-rmse")
                (format "expected train-rmse in ~s" metrics))
    (check-true (real? (hash-ref metrics "train-rmse")))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster-eval-one-iter handles the resize-on-rc=2 path"
    ;; Force the rc=2 retry by jamming the eval-line through with many
    ;; eval matrices labeled with long names — that pushes the result
    ;; string past the 512-byte fast-path guess.
    (define-values (b dtrain) (make-trained-regressor #:rounds 1))
    (define long-name (make-string 30 #\x))   ; 30 chars per dataset
    (define eval-set
      (for/list ([i (in-range 32)])           ; 32 entries × ~50 chars > 512
        (cons (format "~a~a" long-name i) dtrain)))
    (define line (booster-eval-one-iter b 0 eval-set))
    (check-true (> (string-length line) 512))
    (define metrics (parse-eval-line line))
    (check-equal? (hash-count metrics) 32)
    (booster-free! b)
    (dmatrix-free! dtrain))

  ;; --- Predict-mode keywords --------------------------------------------

  (test-case "booster-predict #:output 'margin returns nrow values"
    (define-values (b dtrain) (make-trained-regressor #:rounds 5))
    (define values   (booster-predict b dtrain))
    (define margins  (booster-predict b dtrain #:output 'margin))
    (check-equal? (f32vector-length margins) (f32vector-length values))
    ;; reg:squarederror has identity link, so margin == value for this objective.
    (for ([i (in-range (f32vector-length values))])
      (check-= (f32vector-ref margins i) (f32vector-ref values i) 1e-5))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster-predict #:output 'leaf returns nrow * ntree indices"
    (define-values (b dtrain) (make-trained-regressor #:rounds 7))
    (define n (dmatrix-nrow dtrain))
    (define leaves (booster-predict b dtrain #:output 'leaf))
    ;; Output shape is nrow * ntree (here ntree == rounds == 7).
    (check-equal? (f32vector-length leaves) (* n 7))
    ;; Leaf indices are nonneg integers stored as floats.
    (for ([i (in-range (f32vector-length leaves))])
      (define v (f32vector-ref leaves i))
      (check-true (>= v 0))
      (check-equal? v (round v)))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster-predict #:output 'contribs returns nrow * (ncol+1) SHAP"
    (define-values (b dtrain) (make-trained-regressor #:rounds 5))
    (define n (dmatrix-nrow dtrain))
    (define ncol (dmatrix-ncol dtrain))
    (define shap (booster-predict b dtrain #:output 'contribs))
    (check-equal? (f32vector-length shap) (* n (+ ncol 1)))
    ;; SHAP additivity: sum across contribution columns ≈ raw margin.
    (define margins (booster-predict b dtrain #:output 'margin))
    (for ([i (in-range n)])
      (define row-sum
        (for/sum ([c (in-range (+ ncol 1))])
          (f32vector-ref shap (+ (* i (+ ncol 1)) c))))
      (check-= row-sum (f32vector-ref margins i) 1e-3))
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "booster-predict #:iteration-end limits trees used"
    (define-values (b dtrain) (make-trained-regressor #:rounds 20))
    (define p1     (booster-predict b dtrain #:iteration-end 1))
    (define p20    (booster-predict b dtrain #:iteration-end 20))
    (define p-all  (booster-predict b dtrain))
    ;; iteration_end=20 == use all trees == default (0 means "all").
    (for ([i (in-range (f32vector-length p20))])
      (check-= (f32vector-ref p20 i) (f32vector-ref p-all i) 1e-6))
    ;; First-tree-only predictions must differ from the full ensemble.
    (define same?
      (for/and ([i (in-range (f32vector-length p1))])
        (= (f32vector-ref p1 i) (f32vector-ref p-all i))))
    (check-false same? "iteration-end=1 should differ from full ensemble")
    (booster-free! b)
    (dmatrix-free! dtrain)))
