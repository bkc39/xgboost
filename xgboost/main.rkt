#lang racket/base

(require ffi/vector
         json
         racket/contract
         racket/list
         racket/match
         racket/string
         (prefix-in ffi: "ffi.rkt"))

;; DMatrix and Booster are wrapper structs defined in xgboost/ffi.rkt; the
;; high-level layer re-exports the predicates and accessors so users see one
;; consistent surface regardless of which entry point they require.
(define dmatrix? ffi:dmatrix?)
(define dmatrix-handle ffi:dmatrix-handle)
(define dmatrix-rows ffi:dmatrix-rows)
(define dmatrix-cols ffi:dmatrix-cols)
(define booster? ffi:booster?)
(define booster-handle ffi:booster-handle)
(define booster-cache ffi:booster-cache)

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [xgboost-build-info (-> string?)]
  [xgboost-get-global-config (-> string?)]
  [xgboost-set-global-config! (-> string? void?)]
  [xgboost-register-log-callback! (-> (-> string? any/c) void?)]
  [make-dmatrix
   (->* (any/c)
        (#:nrow (or/c #f exact-positive-integer?)
         #:ncol (or/c #f exact-positive-integer?)
         #:missing real?
         #:labels (or/c #f any/c)
         #:weights (or/c #f any/c))
        dmatrix?)]
  [make-dmatrix-from-csr
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [make-dmatrix-from-csc
   (->* (u64vector? u32vector? f32vector? exact-positive-integer?)
        (real?)
        dmatrix?)]
  [make-dmatrix-from-columnar
   (->* ((listof f32vector?))
        (real?)
        dmatrix?)]
  [make-dmatrix-from-uri
   (->* ((or/c path-string? string?))
        (#:format (or/c #f "libsvm" "csv")
         #:silent? any/c)
        dmatrix?)]
  [dmatrix-rows (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-cols (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix->list (-> dmatrix? (listof (listof real?)))]
  [dmatrix-show (->* (dmatrix?) (output-port?) void?)]
  [dmatrix-slice
   (->* (dmatrix? (or/c list? vector? s32vector?))
        (#:allow-groups? any/c)
        dmatrix?)]
  [dmatrix-save-binary!
   (->* (dmatrix? path-string?) (#:silent? any/c) void?)]
  [dmatrix-set-label! (-> dmatrix? any/c void?)]
  [dmatrix-set-weight! (-> dmatrix? any/c void?)]
  [dmatrix-set-base-margin! (-> dmatrix? any/c void?)]
  [dmatrix-set-label-lower-bound! (-> dmatrix? any/c void?)]
  [dmatrix-set-label-upper-bound! (-> dmatrix? any/c void?)]
  [dmatrix-set-group! (-> dmatrix? any/c void?)]
  [dmatrix-set-feature-names! (-> dmatrix? (listof string?) void?)]
  [dmatrix-set-feature-types! (-> dmatrix? (listof string?) void?)]
  [dmatrix-label (-> dmatrix? (listof real?))]
  [dmatrix-weight (-> dmatrix? (listof real?))]
  [dmatrix-base-margin (-> dmatrix? (listof real?))]
  [dmatrix-group-ptr (-> dmatrix? (listof exact-nonnegative-integer?))]
  [dmatrix-feature-names (-> dmatrix? (listof string?))]
  [dmatrix-feature-types (-> dmatrix? (listof string?))]
  [dmatrix-quantile-cut
   (-> dmatrix? (values (listof exact-nonnegative-integer?) f32vector?))]
  [booster-num-feature (-> booster? exact-nonnegative-integer?)]
  [booster-boosted-rounds (-> booster? exact-nonnegative-integer?)]
  [booster-reset! (-> booster? void?)]
  [booster-slice
   (->* (booster? exact-integer? exact-integer?)
        (exact-positive-integer?)
        booster?)]
  [booster-set-attr! (-> booster? string? string? void?)]
  [booster-attr (-> booster? string? (or/c #f string?))]
  [booster-attr-names (-> booster? (listof string?))]
  [booster-delete-attr! (-> booster? string? void?)]
  [booster-set-feature-names! (-> booster? (listof string?) void?)]
  [booster-set-feature-types! (-> booster? (listof string?) void?)]
  [booster-feature-names (-> booster? (listof string?))]
  [booster-feature-types (-> booster? (listof string?))]
  [booster-config (-> booster? string?)]
  [booster-set-config! (-> booster? string? void?)]
  [booster-dump
   (->* (booster?)
        (#:format (or/c "text" "json" "dot")
         #:with-stats? any/c
         #:feature-names (or/c #f (listof string?))
         #:feature-types (or/c #f (listof string?)))
        (listof string?))]
  [booster-feature-score
   (->* (booster?)
        (#:importance-type string?
         #:feature-names (or/c #f (listof string?))
         #:config (or/c #f string?))
        (hash/c symbol? any/c))]
  [predict-from-dense
   (->* (booster? any/c)
        (#:nrow (or/c #f exact-positive-integer?)
         #:ncol (or/c #f exact-positive-integer?)
         #:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [predict-from-csr
   (->* (booster? u64vector? u32vector? f32vector? exact-positive-integer?)
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [predict-from-columnar
   (->* (booster? (listof f32vector?))
        (#:missing real?
         #:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [train
   (->* (dmatrix?)
        (#:params any/c
         #:rounds exact-nonnegative-integer?
         #:evals (listof (cons/c string? dmatrix?))
         #:objective (or/c #f any/c)
         #:objective-fn (or/c #f (-> f32vector? dmatrix? any))
         #:eta (or/c #f any/c)
         #:max-depth (or/c #f any/c)
         #:num-class (or/c #f any/c)
         #:eval-metric (or/c #f any/c)
         #:verbosity (or/c #f any/c))
        booster?)]
  [booster-set-param! (-> booster? any/c any/c void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-train-one-iter!
   (-> booster? exact-integer? dmatrix? any/c any/c void?)]
  [booster->bytes (-> booster? bytes?)]
  [bytes->booster (-> bytes? booster?)]
  [predict
   (->* (booster? dmatrix?)
        (#:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [save-model (-> booster? path-string? void?)]
  [load-model (-> path-string? booster?)]
  [make-booster (-> booster?)]
  [save-model-to-bytes (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [load-model-from-bytes (-> bytes? booster?)]
  [eval-one-iter (-> booster? exact-integer?
                     (listof (cons/c string? dmatrix?))
                     string?)]
  [booster-cache (-> booster? (listof dmatrix?))]
  [parse-eval-line (-> string? (hash/c string? real?))]
  [cuda-available? (-> boolean?)])
 dmatrix?
 booster?)

(define xgboost-version ffi:xgboost-version)
(define xgboost-build-info ffi:xgboost-build-info)
(define xgboost-get-global-config ffi:xgboost-get-global-config)
(define xgboost-set-global-config! ffi:xgboost-set-global-config!)
(define xgboost-register-log-callback! ffi:xgboost-register-log-callback!)
(define parse-eval-line ffi:parse-eval-line)

(define (cuda-available?)
  (define info (string->jsexpr (ffi:xgboost-build-info)))
  (if (hash-ref info 'USE_CUDA #f) #t #f))

(define (sequence->f32vector who xs)
  (cond
    [(f32vector? xs) xs]
    [(list? xs) (list->f32vector (map exact->inexact xs))]
    [(vector? xs) (list->f32vector (map exact->inexact (vector->list xs)))]
    [else (raise-argument-error who "list, vector, or f32vector" xs)]))

(define (row-sequence? v)
  (or (list? v) (vector? v)))

(define (rows->matrix who rows0)
  (define rows (if (vector? rows0) (vector->list rows0) rows0))
  (unless (and (list? rows) (andmap row-sequence? rows))
    (raise-argument-error who "list-of-lists or vector-of-vectors" rows0))
  (define nrow (length rows))
  (define ncol
    (cond
      [(zero? nrow) (raise-argument-error who "at least one row" rows0)]
      [else (length (if (vector? (car rows)) (vector->list (car rows)) (car rows)))]))
  (when (zero? ncol)
    (raise-argument-error who "at least one column" rows0))
  (define flat
    (for/fold ([acc '()] #:result (reverse acc))
              ([row0 (in-list rows)])
      (define row (if (vector? row0) (vector->list row0) row0))
      (unless (= (length row) ncol)
        (error who "ragged matrix: expected ~a columns, got ~a" ncol (length row)))
      (append (reverse row) acc)))
  (values (list->f32vector (map exact->inexact flat)) nrow ncol))

(define (coerce-matrix data nrow ncol)
  (cond
    [(and (or (list? data) (vector? data))
          (positive? (if (vector? data) (vector-length data) (length data)))
          (row-sequence? (if (vector? data) (vector-ref data 0) (car data))))
     (define-values (vec inferred-nrow inferred-ncol)
       (rows->matrix 'make-dmatrix data))
     (when (and nrow (not (= nrow inferred-nrow)))
       (error 'make-dmatrix "given #:nrow ~a does not match inferred row count ~a"
              nrow inferred-nrow))
     (when (and ncol (not (= ncol inferred-ncol)))
       (error 'make-dmatrix "given #:ncol ~a does not match inferred column count ~a"
              ncol inferred-ncol))
     (values vec inferred-nrow inferred-ncol)]
    [else
     (unless (and nrow ncol)
       (error 'make-dmatrix
              "flat data requires both #:nrow and #:ncol"))
     (values (sequence->f32vector 'make-dmatrix data) nrow ncol)]))

(define (make-dmatrix data
                      #:nrow [nrow #f]
                      #:ncol [ncol #f]
                      #:missing [missing +nan.0]
                      #:labels [labels #f]
                      #:weights [weights #f])
  (define-values (vec rows cols) (coerce-matrix data nrow ncol))
  (define dm (ffi:dmatrix-create-from-mat vec rows cols missing))
  (when labels
    (define label-vec (sequence->f32vector 'make-dmatrix labels))
    (unless (= (f32vector-length label-vec) rows)
      (error 'make-dmatrix "label length ~a does not match row count ~a"
             (f32vector-length label-vec) rows))
    (ffi:dmatrix-set-float-info! dm "label" label-vec))
  (when weights
    (define weight-vec (sequence->f32vector 'make-dmatrix weights))
    (unless (= (f32vector-length weight-vec) rows)
      (error 'make-dmatrix "weight length ~a does not match row count ~a"
             (f32vector-length weight-vec) rows))
    (ffi:dmatrix-set-float-info! dm "weight" weight-vec))
  dm)

(define (make-dmatrix-from-csr indptr indices data ncol [missing +nan.0])
  (ffi:dmatrix-create-from-csr indptr indices data ncol missing))

(define (make-dmatrix-from-csc indptr indices data nrow [missing +nan.0])
  (ffi:dmatrix-create-from-csc indptr indices data nrow missing))

(define (make-dmatrix-from-columnar columns [missing +nan.0])
  (ffi:dmatrix-create-from-columnar columns missing))

(define (make-dmatrix-from-uri uri-or-path
                               #:format [fmt #f]
                               #:silent? [silent? #t])
  (define base
    (cond [(path? uri-or-path) (path->string uri-or-path)]
          [else uri-or-path]))
  (define uri (if fmt (format "~a?format=~a" base fmt) base))
  (define cfg
    (format "{\"uri\":~s,\"silent\":~a}" uri (if silent? 1 0)))
  (ffi:dmatrix-create-from-uri cfg))

(define (dmatrix->list dm)
  (ffi:dmatrix->list dm))

(define (dmatrix-show dm [port (current-output-port)])
  (ffi:dmatrix-show dm port))

(define (dmatrix-slice dm indices #:allow-groups? [allow-groups? #f])
  (ffi:dmatrix-slice dm indices #:allow-groups? allow-groups?))

(define (dmatrix-save-binary! dm path #:silent? [silent? #t])
  (ffi:dmatrix-save-binary! dm path #:silent? silent?))

(define (sequence->u32vector who xs)
  (cond
    [(u32vector? xs) xs]
    [(list? xs) (list->u32vector xs)]
    [(vector? xs) (list->u32vector (vector->list xs))]
    [else (raise-argument-error who "list, vector, or u32vector" xs)]))

(define (set-float-info! who dm field xs)
  (ffi:dmatrix-set-float-info! dm field (sequence->f32vector who xs)))

(define (dmatrix-set-label! dm xs) (set-float-info! 'dmatrix-set-label! dm "label" xs))
(define (dmatrix-set-weight! dm xs) (set-float-info! 'dmatrix-set-weight! dm "weight" xs))
(define (dmatrix-set-base-margin! dm xs)
  (set-float-info! 'dmatrix-set-base-margin! dm "base_margin" xs))

(define (dmatrix-set-label-lower-bound! dm xs)
  (set-float-info! 'dmatrix-set-label-lower-bound! dm "label_lower_bound" xs))

(define (dmatrix-set-label-upper-bound! dm xs)
  (set-float-info! 'dmatrix-set-label-upper-bound! dm "label_upper_bound" xs))

(define (dmatrix-set-group! dm sizes)
  (ffi:dmatrix-set-uint-info! dm "group"
                              (sequence->u32vector 'dmatrix-set-group! sizes)))

(define (dmatrix-set-feature-names! dm names)
  (ffi:dmatrix-set-feature-info! dm "feature_name" names))

(define (dmatrix-set-feature-types! dm types)
  (ffi:dmatrix-set-feature-info! dm "feature_type" types))

(define (get-float-info-list dm field)
  (f32vector->list (ffi:dmatrix-get-float-info dm field)))

(define (dmatrix-label dm) (get-float-info-list dm "label"))
(define (dmatrix-weight dm) (get-float-info-list dm "weight"))
(define (dmatrix-base-margin dm) (get-float-info-list dm "base_margin"))

(define (dmatrix-group-ptr dm)
  (define vec (ffi:dmatrix-get-uint-info dm "group_ptr"))
  (for/list ([i (in-range (u32vector-length vec))])
    (u32vector-ref vec i)))

(define (dmatrix-feature-names dm)
  (ffi:dmatrix-get-feature-info dm "feature_name"))

(define (dmatrix-feature-types dm)
  (ffi:dmatrix-get-feature-info dm "feature_type"))

(define (dmatrix-quantile-cut dm)
  (define-values (indptr-json data-json)
    (ffi:dmatrix-get-quantile-cut dm))
  (define indptr (ffi:array-interface->u64vector indptr-json))
  (define data (ffi:array-interface->f32vector data-json))
  (values (for/list ([i (in-range (u64vector-length indptr))])
            (u64vector-ref indptr i))
          data))

(define (booster-num-feature b)
  (ffi:booster-num-feature b))

(define (booster-boosted-rounds b)
  (ffi:booster-boosted-rounds b))

(define (booster-reset! b)
  (ffi:booster-reset! b))

(define (booster-slice b begin-layer end-layer [step 1])
  (ffi:booster-slice b begin-layer end-layer step))

(define (booster-set-attr! b key value)
  (ffi:booster-set-attr! b key value))

(define (booster-attr b key)
  (ffi:booster-get-attr b key))

(define (booster-attr-names b)
  (ffi:booster-get-attr-names b))

(define (booster-delete-attr! b key)
  (ffi:booster-delete-attr! b key))

(define (booster-set-feature-names! b names)
  (ffi:booster-set-feature-info! b "feature_name" names))

(define (booster-set-feature-types! b types)
  (ffi:booster-set-feature-info! b "feature_type" types))

(define (booster-feature-names b)
  (ffi:booster-get-feature-info b "feature_name"))

(define (booster-feature-types b)
  (ffi:booster-get-feature-info b "feature_type"))

(define (booster-config b)
  (ffi:booster-save-json-config b))

(define (booster-set-config! b cfg)
  (ffi:booster-load-json-config! b cfg))

(define (booster-dump b
                     #:format [fmt "text"]
                     #:with-stats? [stats? #f]
                     #:feature-names [names #f]
                     #:feature-types [types #f])
  (cond
    [(or names types)
     (unless (and names types)
       (error 'booster-dump
              "#:feature-names and #:feature-types must be provided together"))
     (ffi:booster-dump-model-with-features b names types
                                           #:format fmt
                                           #:with-stats? stats?)]
    [else
     (ffi:booster-dump-model b
                             #:format fmt
                             #:with-stats? stats?)]))

(define (booster-feature-score b
                               #:importance-type [importance-type "weight"]
                               #:feature-names [names #f]
                               #:config [config #f])
  (ffi:booster-feature-score b
                             #:importance-type importance-type
                             #:feature-names names
                             #:config config))

(define (preds->shape preds as)
  (case as
    [(f32vector) preds]
    [(list) (f32vector->list preds)]
    [else (raise-argument-error 'predict "'list or 'f32vector" as)]))

(define (predict-from-dense b data
                            #:nrow [nrow #f]
                            #:ncol [ncol #f]
                            #:missing [missing +nan.0]
                            #:output [output 'value]
                            #:iteration-end [iter-end 0]
                            #:as [as 'list])
  (define-values (vec rows cols) (coerce-matrix data nrow ncol))
  (preds->shape
   (ffi:booster-predict-from-dense b vec rows cols
                                   #:missing missing
                                   #:output output
                                   #:iteration-end iter-end)
   as))

(define (predict-from-csr b indptr indices data ncol
                          #:missing [missing +nan.0]
                          #:output [output 'value]
                          #:iteration-end [iter-end 0]
                          #:as [as 'list])
  (preds->shape
   (ffi:booster-predict-from-csr b indptr indices data ncol
                                 #:missing missing
                                 #:output output
                                 #:iteration-end iter-end)
   as))

(define (predict-from-columnar b columns
                               #:missing [missing +nan.0]
                               #:output [output 'value]
                               #:iteration-end [iter-end 0]
                               #:as [as 'list])
  (preds->shape
   (ffi:booster-predict-from-columnar b columns
                                      #:missing missing
                                      #:output output
                                      #:iteration-end iter-end)
   as))

(define (param-key->string k)
  (cond
    [(string? k) k]
    [(symbol? k) (string-replace (symbol->string k) "-" "_")]
    [(keyword? k) (string-replace (keyword->string k) "-" "_")]
    [else (format "~a" k)]))

(define (param-value->string v)
  (cond
    [(string? v) v]
    [(symbol? v) (symbol->string v)]
    [(keyword? v) (keyword->string v)]
    [(boolean? v) (if v "true" "false")]
    [(number? v) (number->string v)]
    [else (format "~a" v)]))

(define (params->pairs params)
  (cond
    [(not params) '()]
    [(hash? params) (hash->list params)]
    [(list? params)
     (for/list ([p (in-list params)])
       (match p
         [(cons k v) (cons k v)]
         [(list k v) (cons k v)]
         [_ (raise-argument-error 'train "hash or alist of parameters" params)]))]
    [else (raise-argument-error 'train "hash or alist of parameters" params)]))

(define (maybe-param key value)
  (if value (list (cons key value)) '()))

(define (train dtrain
               #:params [params '()]
               #:rounds [rounds 10]
               #:evals [evals '()]
               #:objective [objective #f]
               #:objective-fn [objective-fn #f]
               #:eta [eta #f]
               #:max-depth [max-depth #f]
               #:num-class [num-class #f]
               #:eval-metric [eval-metric #f]
               #:verbosity [verbosity #f])
  (define cache (cons dtrain (map cdr evals)))
  (define b (ffi:booster-create cache))
  (define all-params
    (append (params->pairs params)
            (maybe-param 'objective objective)
            (maybe-param 'eta eta)
            (maybe-param 'max_depth max-depth)
            (maybe-param 'num_class num-class)
            (maybe-param 'eval_metric eval-metric)
            (maybe-param 'verbosity verbosity)))
  (for ([p (in-list all-params)])
    (ffi:booster-set-param! b
                            (param-key->string (car p))
                            (param-value->string (cdr p))))
  (cond
    [objective-fn
     (for ([iter (in-range rounds)])
       (define preds
         (ffi:booster-predict b dtrain #:output 'margin))
       (define-values (grad hess) (objective-fn preds dtrain))
       (ffi:booster-train-one-iter! b iter dtrain
                                    (sequence->f32vector 'train grad)
                                    (sequence->f32vector 'train hess)))]
    [else
     (for ([iter (in-range rounds)])
       (ffi:booster-update-one-iter! b iter dtrain))])
  b)

(define (booster-set-param! b key value)
  (ffi:booster-set-param! b
                          (param-key->string key)
                          (param-value->string value)))

(define (booster-update-one-iter! b iter dtrain)
  (ffi:booster-update-one-iter! b iter dtrain))

(define (booster-train-one-iter! b iter dtrain grad hess)
  (ffi:booster-train-one-iter! b iter dtrain
                               (sequence->f32vector 'booster-train-one-iter! grad)
                               (sequence->f32vector 'booster-train-one-iter! hess)))

(define (booster->bytes b)
  (ffi:booster-serialize-to-bytes b))

(define (bytes->booster bs)
  (define b (ffi:booster-create))
  (ffi:booster-unserialize-from-bytes! b bs)
  b)

(define (predict b dmat
                 #:output [output 'value]
                 #:iteration-end [iteration-end 0]
                 #:as [as 'list])
  (define preds
    (ffi:booster-predict b dmat
                         #:output output
                         #:iteration-end iteration-end))
  (case as
    [(f32vector) preds]
    [(list) (f32vector->list preds)]
    [else (raise-argument-error 'predict "'list or 'f32vector" as)]))

(define (save-model b path)
  (ffi:booster-save-model! b path))

(define (load-model path)
  (define b (ffi:booster-create))
  (ffi:booster-load-model! b path)
  b)

(define (make-booster)
  (ffi:booster-create))

(define (save-model-to-bytes b #:format [fmt "ubj"])
  (ffi:booster-save-model-to-bytes b #:format fmt))

(define (load-model-from-bytes bs)
  (define b (ffi:booster-create))
  (ffi:booster-load-model-from-bytes! b bs)
  b)

(define (eval-one-iter b iter evals)
  (ffi:booster-eval-one-iter
   b
   iter
   (for/list ([entry (in-list evals)])
     (cons (car entry) (cdr entry)))))

(module+ main
  (printf "xgboost version: ~a\n" (xgboost-version))
  (define dtrain
    (make-dmatrix '((1.0 2.0 0.5)
                    (2.0 1.0 1.5)
                    (3.0 0.5 0.0)
                    (0.5 3.0 2.0))
                  #:labels '(3.5 3.5 6.5 2.0)))
  (define b
    (train dtrain
           #:objective "reg:squarederror"
           #:max-depth 2
           #:eta 0.2
           #:verbosity 0
           #:rounds 5))
  (printf "first prediction: ~a\n" (car (predict b dtrain))))

(module+ test
  (require rackunit
           racket/file
           "ffi/raw.rkt")

  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))
  (check-regexp-match #rx"^\\{" (xgboost-build-info))
  (check-true (procedure? xgb-version/raw))
  (check-true (procedure? ffi:xgboost-version))
  (check-pred boolean? (cuda-available?))

  (test-case "make-dmatrix accepts list rows and labels"
    (define dm (make-dmatrix '((1 2) (3 4) (5 6)) #:labels '(1 2 3)))
    (check-pred dmatrix? dm)
    (check-equal? (ffi:dmatrix-nrow dm) 3)
    (check-equal? (ffi:dmatrix-ncol dm) 2)
    (check-equal? (f32vector->list (ffi:dmatrix-get-float-info dm "label"))
                  '(1.0 2.0 3.0)))

  (test-case "make-dmatrix accepts vector rows, flat vectors, and f32vectors"
    (define dm1 (make-dmatrix (vector (vector 1 2) (vector 3 4))
                              #:weights (vector 1.0 0.5)))
    (check-equal? (ffi:dmatrix-nrow dm1) 2)
    (check-equal? (ffi:dmatrix-ncol dm1) 2)
    (define dm2 (make-dmatrix (vector 1 2 3 4 5 6) #:nrow 2 #:ncol 3))
    (check-equal? (ffi:dmatrix-nrow dm2) 2)
    (check-equal? (ffi:dmatrix-ncol dm2) 3)
    (define dm3 (make-dmatrix (f32vector 1.0 2.0 3.0 4.0) #:nrow 4 #:ncol 1))
    (check-equal? (ffi:dmatrix-nrow dm3) 4)
    (check-equal? (ffi:dmatrix-ncol dm3) 1))

  (test-case "train and predict regression"
    (define dm
      (make-dmatrix '((1.0 2.0 0.5)
                      (2.0 1.0 1.5)
                      (3.0 0.5 0.0)
                      (0.5 3.0 2.0)
                      (4.0 2.0 1.0)
                      (1.5 1.5 0.5)
                      (2.5 3.5 1.5)
                      (0.0 1.0 0.0))
                    #:labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
    (define b (train dm
                     #:params '((objective . "reg:squarederror")
                                (max_depth . 3)
                                (eta . 0.1)
                                (verbosity . 0))
                     #:rounds 20))
    (define preds (predict b dm))
    (check-equal? (length preds) 8)
    (check-true (andmap real? preds))
    (define line (eval-one-iter b 19 (list (cons "train" dm))))
    (check-true (hash-has-key? (parse-eval-line line) "train-rmse"))
    (check-equal? (booster-cache b) (list dm)))

  (test-case "binary classification probabilities"
    (define dm
      (make-dmatrix '((0 0) (0 1) (1 0) (1 1) (2 1) (2 2))
                    #:labels '(0 0 0 1 1 1)))
    (define b (train dm
                     #:objective "binary:logistic"
                     #:eval-metric "logloss"
                     #:max-depth 2
                     #:eta 0.3
                     #:verbosity 0
                     #:rounds 10))
    (for ([p (in-list (predict b dm))])
      (check-true (<= 0.0 p 1.0))))

  (test-case "multiclass softprob returns rows * classes predictions"
    (define dm
      (make-dmatrix '((0 0) (0 1) (1 0) (3 3) (3 4) (4 3))
                    #:labels '(0 0 0 1 1 1)))
    (define b (train dm
                     #:objective "multi:softprob"
                     #:num-class 2
                     #:max-depth 2
                     #:eta 0.2
                     #:verbosity 0
                     #:rounds 5))
    (check-equal? (length (predict b dm)) 12))

  (test-case "save and load model"
    (define dm (make-dmatrix '((1) (2) (3) (4)) #:labels '(1 2 3 4)))
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:verbosity 0
                     #:rounds 5))
    (define base (predict b dm))
    (define tmp (make-temporary-file "xgboost-~a.json"))
    (dynamic-wind
      void
      (lambda ()
        (save-model b tmp)
        (define b2 (load-model tmp))
        (check-equal? (predict b2 dm) base))
      (lambda () (when (file-exists? tmp) (delete-file tmp))))
    (define b3 (load-model-from-bytes (save-model-to-bytes b)))
    (check-equal? (predict b3 dm) base))

  (test-case "make-dmatrix-from-csr / csc / columnar"
    (define csr (make-dmatrix-from-csr
                 (u64vector 0 2 4)
                 (u32vector 0 2 1 2)
                 (f32vector 1.0 3.0 5.0 6.0)
                 3
                 -1.0))
    (check-equal? (dmatrix-rows csr) 2)
    (check-equal? (dmatrix-cols csr) 3)
    (define csc (make-dmatrix-from-csc
                 (u64vector 0 1 2 4)
                 (u32vector 0 1 0 1)
                 (f32vector 1.0 5.0 3.0 6.0)
                 2
                 -1.0))
    (check-equal? (dmatrix-rows csc) 2)
    (check-equal? (dmatrix-cols csc) 3)
    (define cols (make-dmatrix-from-columnar
                  (list (f32vector 1.0 4.0)
                        (f32vector 2.0 5.0)
                        (f32vector 3.0 6.0))))
    (check-equal? (dmatrix-rows cols) 2)
    (check-equal? (dmatrix-cols cols) 3)
    (check-equal? (dmatrix->list cols)
                  '((1.0 2.0 3.0) (4.0 5.0 6.0))))

  (test-case "dmatrix-slice and dmatrix-save-binary! round-trip via URI"
    (define dm
      (make-dmatrix '((1.0 2.0) (3.0 4.0) (5.0 6.0))))
    (define sliced (dmatrix-slice dm '(2 0)))
    (check-equal? (dmatrix->list sliced) '((5.0 6.0) (1.0 2.0)))
    (define tmp (make-temporary-file "xgboost-slice-~a.buffer"))
    (when (file-exists? tmp) (delete-file tmp))
    (dynamic-wind
      void
      (lambda ()
        (dmatrix-save-binary! sliced tmp)
        (define loaded (make-dmatrix-from-uri tmp))
        (check-equal? (dmatrix->list loaded) '((5.0 6.0) (1.0 2.0)))
        (check-equal? (dmatrix-rows loaded) 2)
        (check-equal? (dmatrix-cols loaded) 2))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))

  (test-case "make-dmatrix-from-uri loads libsvm with #:format"
    (define tmp (make-temporary-file "xgboost-libsvm-~a.txt"))
    (dynamic-wind
      void
      (lambda ()
        (with-output-to-file tmp
          (lambda ()
            (displayln "0 0:1 1:2 2:3")
            (displayln "1 0:4 1:5 2:6"))
          #:exists 'truncate)
        (define dm (make-dmatrix-from-uri tmp #:format "libsvm"))
        (check-equal? (dmatrix-rows dm) 2)
        (check-equal? (dmatrix-cols dm) 3))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))

  (test-case "dmatrix metadata: named info and feature info round-trip"
    (define dm (make-dmatrix '((1.0 2.0) (3.0 4.0))))
    (dmatrix-set-label! dm '(0.25 0.75))
    (dmatrix-set-weight! dm (vector 1.0 2.0))
    (dmatrix-set-base-margin! dm (f32vector 0.1 0.2))
    (dmatrix-set-group! dm '(2))
    (dmatrix-set-feature-names! dm '("height" "weight"))
    (dmatrix-set-feature-types! dm '("q" "q"))
    (check-equal? (dmatrix-label dm) '(0.25 0.75))
    (check-equal? (dmatrix-weight dm) '(1.0 2.0))
    (define margin (dmatrix-base-margin dm))
    (check-equal? (length margin) 2)
    (check-true (< (abs (- (first margin) 0.1)) 1e-6))
    (check-true (< (abs (- (second margin) 0.2)) 1e-6))
    (check-equal? (dmatrix-group-ptr dm) '(0 2))
    (check-equal? (dmatrix-feature-names dm) '("height" "weight"))
    (check-equal? (dmatrix-feature-types dm) '("q" "q")))

  (test-case "dmatrix-quantile-cut returns parsed Racket data"
    (define dm
      (make-dmatrix '((1.0 2.0) (3.0 4.0) (5.0 6.0) (7.0 8.0))
                    #:labels '(1.0 3.0 5.0 7.0)))
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:params '((tree_method . "hist"))
                     #:max-depth 2
                     #:verbosity 0
                     #:rounds 1))
    ;; suppress unused-var warnings for booster — `train` registered the
    ;; quantile state on `dm` via its hist-mode sketch.
    (void b)
    (define-values (indptr data) (dmatrix-quantile-cut dm))
    (check-true (pair? indptr))
    (check-equal? (car indptr) 0)
    (check-true (positive? (last indptr)))
    (check-true (f32vector? data))
    (check-equal? (f32vector-length data) (last indptr)))

  (test-case "booster lifecycle: slice, config, reset, num-feature, rounds"
    (define dm
      (make-dmatrix '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
                      (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0))
                    #:labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:max-depth 3
                     #:eta 0.1
                     #:verbosity 0
                     #:rounds 8))
    (check-equal? (booster-boosted-rounds b) 8)
    (check-equal? (booster-num-feature b) 3)
    (define sliced (booster-slice b 0 3))
    (check-equal? (booster-boosted-rounds sliced) 3)
    (define cfg (booster-config b))
    (check-true (regexp-match? #rx"^\\{" cfg))
    (booster-reset! b)
    (check-equal? (booster-boosted-rounds b) 8))

  (test-case "booster attrs round-trip"
    (define b (make-booster))
    (booster-set-attr! b "owner" "racket")
    (booster-set-attr! b "purpose" "example")
    (check-equal? (booster-attr b "owner") "racket")
    (check-equal? (booster-attr b "purpose") "example")
    (check-equal? (sort (booster-attr-names b) string<?) '("owner" "purpose"))
    (booster-delete-attr! b "purpose")
    (check-false (booster-attr b "purpose"))
    (check-equal? (booster-attr-names b) '("owner")))

  (test-case "booster feature info, dump, and feature score"
    (define names '("x0" "x1" "x2"))
    (define types '("q" "q" "q"))
    (define dm
      (make-dmatrix '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
                      (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0))
                    #:labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
    (dmatrix-set-feature-names! dm names)
    (dmatrix-set-feature-types! dm types)
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:max-depth 3
                     #:eta 0.1
                     #:verbosity 0
                     #:rounds 20))
    (booster-set-feature-names! b names)
    (booster-set-feature-types! b types)
    (check-equal? (booster-feature-names b) names)
    (check-equal? (booster-feature-types b) types)
    (check-true (positive? (length (booster-dump b))))
    (define json-dump (booster-dump b #:format "json"))
    (check-true (regexp-match? #rx"\\{" (car json-dump)))
    (define named-dump (booster-dump b #:feature-names names #:feature-types types))
    (check-true (ormap (lambda (s) (regexp-match? #rx"x[0-2]" s)) named-dump))
    (define score (booster-feature-score b #:feature-names names))
    (check-true (pair? (hash-ref score 'features)))
    (check-true (andmap positive? (f32vector->list (hash-ref score 'scores)))))

  (test-case "inplace predict variants match DMatrix predict"
    (define rows
      '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
        (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0)))
    (define dm (make-dmatrix rows #:labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:max-depth 3
                     #:eta 0.1
                     #:verbosity 0
                     #:rounds 20))
    (define base (predict b dm #:as 'f32vector))
    (define dense
      (predict-from-dense b rows #:missing -1.0 #:as 'f32vector))
    (define columnar
      (predict-from-columnar b
                             (list (f32vector 1.0 2.0 3.0 0.5 4.0 1.5 2.5 0.0)
                                   (f32vector 2.0 1.0 0.5 3.0 2.0 1.5 3.5 1.0)
                                   (f32vector 0.5 1.5 0.0 2.0 1.0 0.5 1.5 0.0))
                             #:missing -1.0
                             #:as 'f32vector))
    (define csr
      (predict-from-csr b
                        (u64vector 0 3 6 9 12 15 18 21 24)
                        (u32vector 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
                        (f32vector 1.0 2.0 0.5 2.0 1.0 1.5 3.0 0.5 0.0
                                   0.5 3.0 2.0 4.0 2.0 1.0 1.5 1.5 0.5
                                   2.5 3.5 1.5 0.0 1.0 0.0)
                        3
                        #:missing -1.0
                        #:as 'f32vector))
    (define (close? a b)
      (and (= (f32vector-length a) (f32vector-length b))
           (for/and ([i (in-range (f32vector-length a))])
             (< (abs (- (f32vector-ref a i) (f32vector-ref b i))) 1e-6))))
    (check-true (close? dense base))
    (check-true (close? columnar base))
    (check-true (close? csr base)))

  (test-case "train with #:objective-fn improves MSE"
    (define labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
    (define dm
      (make-dmatrix '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
                      (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0))
                    #:labels labels))
    (define label-vec (list->f32vector (map exact->inexact labels)))
    (define (squared-error preds _dtrain)
      (define n (f32vector-length preds))
      (define grad (make-f32vector n))
      (define hess (make-f32vector n 1.0))
      (for ([i (in-range n)])
        (f32vector-set! grad i (- (f32vector-ref preds i)
                                  (f32vector-ref label-vec i))))
      (values grad hess))
    (define (mse preds)
      (/ (for/sum ([i (in-range (f32vector-length preds))])
           (define d (- (f32vector-ref preds i) (f32vector-ref label-vec i)))
           (* d d))
         (f32vector-length preds)))
    (define b (train dm
                     #:objective-fn squared-error
                     #:max-depth 3
                     #:eta 0.2
                     #:verbosity 0
                     #:rounds 20))
    (check-true (< (mse (predict b dm #:as 'f32vector)) 3.0)))

  (test-case "booster->bytes / bytes->booster snapshot resumes training"
    (define dm
      (make-dmatrix '((1.0 2.0 0.5) (2.0 1.0 1.5) (3.0 0.5 0.0) (0.5 3.0 2.0)
                      (4.0 2.0 1.0) (1.5 1.5 0.5) (2.5 3.5 1.5) (0.0 1.0 0.0))
                    #:labels '(3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0)))
    (define b (train dm
                     #:objective "reg:squarederror"
                     #:max-depth 3
                     #:eta 0.1
                     #:verbosity 0
                     #:rounds 5))
    (define snapshot (booster->bytes b))
    (define restored (bytes->booster snapshot))
    (check-equal? (predict restored dm) (predict b dm))))
