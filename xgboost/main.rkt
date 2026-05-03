#lang racket/base

(require ffi/vector
         racket/contract
         racket/list
         racket/match
         racket/string
         (prefix-in ffi: "ffi.rkt"))

(struct dmatrix (handle rows cols)
  #:property prop:custom-write
  (lambda (dm port mode)
    (fprintf port "#<dmatrix:~ax~a>" (dmatrix-rows dm) (dmatrix-cols dm))))

(struct booster (handle cache)
  #:property prop:custom-write
  (lambda (_b port mode)
    (fprintf port "#<booster>")))

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [make-dmatrix
   (->* (any/c)
        (#:nrow (or/c #f exact-positive-integer?)
         #:ncol (or/c #f exact-positive-integer?)
         #:missing real?
         #:labels (or/c #f any/c)
         #:weights (or/c #f any/c))
        dmatrix?)]
  [train
   (->* (dmatrix?)
        (#:params any/c
         #:rounds exact-nonnegative-integer?
         #:evals (listof (cons/c string? dmatrix?))
         #:objective (or/c #f any/c)
         #:eta (or/c #f any/c)
         #:max-depth (or/c #f any/c)
         #:num-class (or/c #f any/c)
         #:eval-metric (or/c #f any/c)
         #:verbosity (or/c #f any/c))
        booster?)]
  [predict
   (->* (booster? dmatrix?)
        (#:output (or/c 'value 'margin 'contribs 'approx-contribs
                        'interactions 'approx-interactions 'leaf)
         #:iteration-end exact-nonnegative-integer?
         #:as (or/c 'list 'f32vector))
        (or/c (listof real?) f32vector?))]
  [save-model (-> booster? path-string? void?)]
  [load-model (-> path-string? booster?)]
  [save-model-to-bytes (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [load-model-from-bytes (-> bytes? booster?)]
  [eval-one-iter (-> booster? exact-integer?
                     (listof (cons/c string? dmatrix?))
                     string?)]
  [parse-eval-line (-> string? (hash/c string? real?))])
 dmatrix?
 booster?)

(define xgboost-version ffi:xgboost-version)
(define parse-eval-line ffi:parse-eval-line)

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
  (define h (ffi:dmatrix-create-from-mat vec rows cols missing))
  (when labels
    (define label-vec (sequence->f32vector 'make-dmatrix labels))
    (unless (= (f32vector-length label-vec) rows)
      (error 'make-dmatrix "label length ~a does not match row count ~a"
             (f32vector-length label-vec) rows))
    (ffi:dmatrix-set-float-info! h "label" label-vec))
  (when weights
    (define weight-vec (sequence->f32vector 'make-dmatrix weights))
    (unless (= (f32vector-length weight-vec) rows)
      (error 'make-dmatrix "weight length ~a does not match row count ~a"
             (f32vector-length weight-vec) rows))
    (ffi:dmatrix-set-float-info! h "weight" weight-vec))
  (dmatrix h rows cols))

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
               #:eta [eta #f]
               #:max-depth [max-depth #f]
               #:num-class [num-class #f]
               #:eval-metric [eval-metric #f]
               #:verbosity [verbosity #f])
  (define cache (cons dtrain (map cdr evals)))
  (define h (ffi:booster-create (map dmatrix-handle cache)))
  (define all-params
    (append (params->pairs params)
            (maybe-param 'objective objective)
            (maybe-param 'eta eta)
            (maybe-param 'max_depth max-depth)
            (maybe-param 'num_class num-class)
            (maybe-param 'eval_metric eval-metric)
            (maybe-param 'verbosity verbosity)))
  (for ([p (in-list all-params)])
    (ffi:booster-set-param! h
                            (param-key->string (car p))
                            (param-value->string (cdr p))))
  (for ([iter (in-range rounds)])
    (ffi:booster-update-one-iter! h iter (dmatrix-handle dtrain)))
  (booster h cache))

(define (predict b dmat
                 #:output [output 'value]
                 #:iteration-end [iteration-end 0]
                 #:as [as 'list])
  (define preds
    (ffi:booster-predict (booster-handle b) (dmatrix-handle dmat)
                         #:output output
                         #:iteration-end iteration-end))
  (case as
    [(f32vector) preds]
    [(list) (f32vector->list preds)]
    [else (raise-argument-error 'predict "'list or 'f32vector" as)]))

(define (save-model b path)
  (ffi:booster-save-model! (booster-handle b) path))

(define (load-model path)
  (define h (ffi:booster-create))
  (ffi:booster-load-model! h path)
  (booster h '()))

(define (save-model-to-bytes b #:format [fmt "ubj"])
  (ffi:booster-save-model-to-bytes (booster-handle b) #:format fmt))

(define (load-model-from-bytes bs)
  (define h (ffi:booster-create))
  (ffi:booster-load-model-from-bytes! h bs)
  (booster h '()))

(define (eval-one-iter b iter evals)
  (ffi:booster-eval-one-iter
   (booster-handle b)
   iter
   (for/list ([entry (in-list evals)])
     (cons (car entry) (dmatrix-handle (cdr entry))))))

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
  (check-true (procedure? xgb-version/raw))
  (check-true (procedure? ffi:xgboost-version))

  (test-case "make-dmatrix accepts list rows and labels"
    (define dm (make-dmatrix '((1 2) (3 4) (5 6)) #:labels '(1 2 3)))
    (check-pred dmatrix? dm)
    (check-equal? (ffi:dmatrix-nrow (dmatrix-handle dm)) 3)
    (check-equal? (ffi:dmatrix-ncol (dmatrix-handle dm)) 2)
    (check-equal? (f32vector->list (ffi:dmatrix-get-float-info (dmatrix-handle dm) "label"))
                  '(1.0 2.0 3.0))
    (ffi:dmatrix-free! (dmatrix-handle dm)))

  (test-case "make-dmatrix accepts vector rows, flat vectors, and f32vectors"
    (define dm1 (make-dmatrix (vector (vector 1 2) (vector 3 4))
                              #:weights (vector 1.0 0.5)))
    (check-equal? (ffi:dmatrix-nrow (dmatrix-handle dm1)) 2)
    (check-equal? (ffi:dmatrix-ncol (dmatrix-handle dm1)) 2)
    (define dm2 (make-dmatrix (vector 1 2 3 4 5 6) #:nrow 2 #:ncol 3))
    (check-equal? (ffi:dmatrix-nrow (dmatrix-handle dm2)) 2)
    (check-equal? (ffi:dmatrix-ncol (dmatrix-handle dm2)) 3)
    (define dm3 (make-dmatrix (f32vector 1.0 2.0 3.0 4.0) #:nrow 4 #:ncol 1))
    (check-equal? (ffi:dmatrix-nrow (dmatrix-handle dm3)) 4)
    (check-equal? (ffi:dmatrix-ncol (dmatrix-handle dm3)) 1)
    (ffi:dmatrix-free! (dmatrix-handle dm1))
    (ffi:dmatrix-free! (dmatrix-handle dm2))
    (ffi:dmatrix-free! (dmatrix-handle dm3)))

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
    (check-equal? (booster-cache b) (list dm))
    (ffi:booster-free! (booster-handle b))
    (ffi:dmatrix-free! (dmatrix-handle dm)))

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
      (check-true (<= 0.0 p 1.0)))
    (ffi:booster-free! (booster-handle b))
    (ffi:dmatrix-free! (dmatrix-handle dm)))

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
    (check-equal? (length (predict b dm)) 12)
    (ffi:booster-free! (booster-handle b))
    (ffi:dmatrix-free! (dmatrix-handle dm)))

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
        (check-equal? (predict b2 dm) base)
        (ffi:booster-free! (booster-handle b2)))
      (lambda () (when (file-exists? tmp) (delete-file tmp))))
    (define b3 (load-model-from-bytes (save-model-to-bytes b)))
    (check-equal? (predict b3 dm) base)
    (ffi:booster-free! (booster-handle b3))
    (ffi:booster-free! (booster-handle b))
    (ffi:dmatrix-free! (dmatrix-handle dm))))
