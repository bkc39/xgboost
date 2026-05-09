#lang racket/base

(require ffi/vector
         xgboost/ffi)

(provide run-example)

;; Demonstrates booster_serialize_to_buffer / unserialize_from_buffer:
;; a full-state snapshot (training caches included) lets a fresh booster
;; resume update_one_iter calls in lockstep with the original.

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
  (define dtrain (dmatrix-create-from-mat features 8 3 -1.0))
  (dmatrix-set-float-info! dtrain "label" labels)
  (define b (booster-create (list dtrain)))
  (booster-set-param! b "objective" "reg:squarederror")
  (booster-set-param! b "max_depth" "3")
  (booster-set-param! b "eta" "0.1")
  (booster-set-param! b "verbosity" "0")
  (train-rounds! b dtrain 0 5)

  ;; Snapshot the partially-trained booster.
  (define snapshot (booster-serialize-to-bytes b))
  (define snapshot-preds (booster-predict b dtrain))

  ;; Restore into a fresh handle.  Predictions match immediately.
  (define restored (booster-create))
  (booster-unserialize-from-bytes! restored snapshot)
  (define restored-preds (booster-predict restored dtrain))

  ;; Continue training from round 5 on both — they must stay in lockstep
  ;; because the snapshot includes XGBoost's internal training caches.
  (train-rounds! b dtrain 5 5)
  (train-rounds! restored dtrain 5 5)

  (define result
    (hash 'snapshot-bytes (bytes-length snapshot)
          'matches-immediately?
          (equal? (f32vector->list snapshot-preds)
                  (f32vector->list restored-preds))
          'matches-after-resume?
          (equal? (f32vector->list (booster-predict b dtrain))
                  (f32vector->list (booster-predict restored dtrain)))))

  (booster-free! restored)
  (booster-free! b)
  (dmatrix-free! dtrain)
  result)

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
