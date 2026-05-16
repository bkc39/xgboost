#lang racket/base

;; Integration tests for the high-level `xgboost` (core) API: DMatrix
;; construction, training, prediction, persistence, and metadata.

(require rackunit
         racket/file
         (only-in racket/list first second last)
         ffi/vector
         (prefix-in ffi: "../foreign.rkt")
         "../foreign/raw.rkt"
         "../main.rkt")

(module+ test
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
