#lang racket/base

;; High-level training.
;;
;; `train` builds a booster, applies parameters (from a hash/alist plus
;; convenience keywords), and runs `#:rounds` boosting iterations — either
;; the built-in objective or a Racket `#:objective-fn` supplying gradient and
;; Hessian.  The remaining procedures are thin per-step training calls.

(require racket/match
         racket/string
         (prefix-in foreign: "../foreign.rkt")
         "coerce.rkt")

(provide train
         booster-set-param!
         booster-update-one-iter!
         booster-train-one-iter!
         eval-one-iter)

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
  (define b (foreign:booster-create cache))
  (define all-params
    (append (params->pairs params)
            (maybe-param 'objective objective)
            (maybe-param 'eta eta)
            (maybe-param 'max_depth max-depth)
            (maybe-param 'num_class num-class)
            (maybe-param 'eval_metric eval-metric)
            (maybe-param 'verbosity verbosity)))
  (for ([p (in-list all-params)])
    (foreign:booster-set-param! b
                                (param-key->string (car p))
                                (param-value->string (cdr p))))
  (cond
    [objective-fn
     (for ([iter (in-range rounds)])
       (define preds
         (foreign:booster-predict b dtrain #:output 'margin))
       (define-values (grad hess) (objective-fn preds dtrain))
       (foreign:booster-train-one-iter! b iter dtrain
                                        (sequence->f32vector 'train grad)
                                        (sequence->f32vector 'train hess)))]
    [else
     (for ([iter (in-range rounds)])
       (foreign:booster-update-one-iter! b iter dtrain))])
  b)

(define (booster-set-param! b key value)
  (foreign:booster-set-param! b
                              (param-key->string key)
                              (param-value->string value)))

(define (booster-update-one-iter! b iter dtrain)
  (foreign:booster-update-one-iter! b iter dtrain))

(define (booster-train-one-iter! b iter dtrain grad hess)
  (foreign:booster-train-one-iter! b iter dtrain
                                   (sequence->f32vector 'booster-train-one-iter! grad)
                                   (sequence->f32vector 'booster-train-one-iter! hess)))

(define (eval-one-iter b iter evals)
  (foreign:booster-eval-one-iter
   b
   iter
   (for/list ([entry (in-list evals)])
     (cons (car entry) (cdr entry)))))
