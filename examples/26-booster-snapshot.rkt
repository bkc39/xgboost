#lang racket/base

(require ffi/vector
         xgboost)

(provide run-example)

;; Demonstrates booster->bytes / bytes->booster: a full-state snapshot
;; (training caches included) lets a fresh booster resume per-iter
;; updates in lockstep with the original.

(define features
  (f32vector 1.0 2.0 0.5
             2.0 1.0 1.5
             3.0 0.5 0.0
             0.5 3.0 2.0
             4.0 2.0 1.0
             1.5 1.5 0.5
             2.5 3.5 1.5
             0.0 1.0 0.0))

(define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define (train-rounds! b dtrain start rounds)
  (for ([iter (in-range start (+ start rounds))])
    (booster-update-one-iter! b iter dtrain)))

(define (run-example)
  (define dtrain
    (make-dmatrix features #:nrow 8 #:ncol 3 #:missing -1.0 #:labels labels))
  ;; Set up the booster with params; train manually to leave per-iter
  ;; control to the example so we can resume after the snapshot.
  (define b
    (train dtrain
           #:objective "reg:squarederror"
           #:max-depth 3
           #:eta 0.1
           #:verbosity 0
           #:rounds 0))
  (train-rounds! b dtrain 0 5)

  ;; Snapshot the partially-trained booster.
  (define snapshot (booster->bytes b))
  (define snapshot-preds (predict b dtrain #:as 'f32vector))

  ;; Restore into a fresh handle.  Predictions match immediately.
  (define restored (bytes->booster snapshot))
  (define restored-preds (predict restored dtrain #:as 'f32vector))

  ;; Continue training from round 5 on both — they must stay in lockstep
  ;; because the snapshot includes XGBoost's internal training caches.
  (train-rounds! b dtrain 5 5)
  (train-rounds! restored dtrain 5 5)

  (hash 'snapshot-bytes (bytes-length snapshot)
        'matches-immediately?
        (equal? (f32vector->list snapshot-preds)
                (f32vector->list restored-preds))
        'matches-after-resume?
        (equal? (f32vector->list (predict b dtrain #:as 'f32vector))
                (f32vector->list (predict restored dtrain #:as 'f32vector)))))

(module+ main
  (define result (run-example))
  (printf "snapshot size: ~a bytes\n" (hash-ref result 'snapshot-bytes))
  (printf "matches immediately?     ~a\n"
          (hash-ref result 'matches-immediately?))
  (printf "matches after resume?    ~a\n"
          (hash-ref result 'matches-after-resume?)))

(module+ test
  (require rackunit)
  (define result (run-example))
  (check-true (> (hash-ref result 'snapshot-bytes) 0))
  (check-true (hash-ref result 'matches-immediately?))
  (check-true (hash-ref result 'matches-after-resume?)))
