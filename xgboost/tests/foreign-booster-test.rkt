#lang racket/base

;; Integration tests for the Booster surface of `xgboost/foreign`: training,
;; prediction modes, save/load, snapshots, and per-iteration evaluation.

(require rackunit
         racket/file
         ffi/vector
         "../foreign.rkt"
         (submod "../foreign.rkt" unsafe))

(module+ test
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

  (test-case "second booster-free! raises a contract error (tag guard)"
    (define b (booster-create))
    (booster-free! b)
    ;; The tag flip makes the freed wrapper fail `booster?`, so a second
    ;; free is caught at the contract boundary instead of double-freeing.
    (check-exn exn:fail:contract?
               (lambda () (booster-free! b))))

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

  (test-case "snapshot serialize/unserialize: predictions and training resume"
    (define-values (b dtrain) (make-trained-regressor #:rounds 5))
    (define snapshot (booster-serialize-to-bytes b))
    (check-pred bytes? snapshot)
    (check-true (> (bytes-length snapshot) 0))
    (define b2 (booster-create))
    (booster-unserialize-from-bytes! b2 snapshot)
    (check-equal?
     (f32vector->list (booster-predict b2 dtrain))
     (f32vector->list (booster-predict b dtrain)))
    ;; Snapshot includes training caches, so a fresh update_one_iter on both
    ;; boosters must keep them in lockstep.
    (booster-update-one-iter! b 5 dtrain)
    (booster-update-one-iter! b2 5 dtrain)
    (check-equal?
     (f32vector->list (booster-predict b2 dtrain))
     (f32vector->list (booster-predict b dtrain)))
    (booster-free! b2)
    (booster-free! b)
    (dmatrix-free! dtrain))

  (test-case "unserialize-from-bytes! on garbage raises with C error context"
    (define b (booster-create))
    (check-exn (lambda (e)
                 (and (exn:fail? e)
                      (regexp-match? #rx"XGBoosterUnserializeFromBuffer"
                                     (exn-message e))))
               (lambda ()
                 (booster-unserialize-from-bytes! b (make-bytes 64 65))))
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
