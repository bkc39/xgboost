#lang racket/base

;; Booster prediction.
;;
;; `booster-predict` runs against an existing DMatrix; the `*-from-dense`,
;; `*-from-csr`, and `*-from-columnar` variants are inplace-prediction APIs
;; that take buffers directly without constructing a DMatrix.
;;
;; XGBoost's predict API requires a JSON config with `type` set; "{}" raises
;; a fatal "Argument `type` is required".  All entry points here default to
;; inference-mode predictions over all iterations.

(require ffi/unsafe
         ffi/vector
         racket/string
         "../raw/booster.rkt"
         "../structs.rkt"
         "../error.rkt"
         "../array-interface.rkt"
         "../dmatrix/ops.rkt")

(provide booster-predict
         booster-predict-from-dense
         booster-predict-from-csr
         booster-predict-from-columnar)

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
