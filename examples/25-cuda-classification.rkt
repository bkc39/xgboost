#lang racket/base

;; CUDA-accelerated binary classification training.
;;
;; Trains the same synthetic dataset as examples/02-train-classifier.rkt but
;; with device=cuda and tree_method=hist, delegating tree construction to the GPU.
;;
;; Requires a CUDA-enabled libxgbcompat (built with ./scripts/build-so.sh linux-cuda
;; or nix build .#cpp-cuda) and a physical NVIDIA GPU at runtime.
;;
;; Run from the repo root:
;;   nix develop .#cuda --command racket examples/25-cuda-classification.rkt

(require ffi/vector
         json
         racket/format
         racket/port
         xgboost/ffi)

(provide run-example cuda-available?)

(define (cuda-available?)
  (define info (with-input-from-string (xgboost-build-info) read-json))
  (hash-ref info 'USE_CUDA #f))

(define features
  (f32vector 0.1 0.2 0.1 0.0
             5.0 4.0 5.5 6.0
             0.3 0.5 0.1 0.2
             4.5 5.0 4.0 5.5
             0.0 0.1 0.2 0.0
             6.0 5.5 6.5 5.0
             0.4 0.3 0.2 0.5
             5.5 6.0 4.5 5.0
             0.2 0.1 0.3 0.1
             4.0 4.5 5.0 4.0))

(define labels (f32vector 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))

(define (run-example)
  (define dtrain (dmatrix-create-from-mat features 10 4))
  (dmatrix-set-float-info! dtrain "label" labels)

  (define booster (booster-create (list dtrain)))
  (booster-set-param! booster "device"      "cuda")
  (booster-set-param! booster "tree_method" "hist")
  (booster-set-param! booster "objective"   "binary:logistic")
  (booster-set-param! booster "max_depth"   "3")
  (booster-set-param! booster "eta"         "0.3")
  (booster-set-param! booster "verbosity"   "0")

  (define rounds 30)
  (for ([iter (in-range rounds)])
    (booster-update-one-iter! booster iter dtrain))

  (define probs (booster-predict booster dtrain))
  (define n (f32vector-length labels))

  (define all-valid-probs?
    (for/and ([i (in-range n)])
      (<= 0.0 (f32vector-ref probs i) 1.0)))

  (define correct
    (for/sum ([i (in-range n)])
      (define truth (inexact->exact (round (f32vector-ref labels i))))
      (define pred  (if (> (f32vector-ref probs i) 0.5) 1 0))
      (if (= pred truth) 1 0)))

  (define result
    (hash 'prediction-count  n
          'all-valid-probs?  all-valid-probs?
          'correct           correct
          'accuracy          (/ correct n)
          'perfect?          (= correct n)))

  (booster-free! booster)
  (dmatrix-free! dtrain)
  result)

(module+ main
  (cond
    [(cuda-available?)
     (define result (run-example))
     (printf "CUDA classification predictions: ~a\n"
             (hash-ref result 'prediction-count))
     (printf "accuracy: ~a/~a (~a%%)\n"
             (hash-ref result 'correct)
             (hash-ref result 'prediction-count)
             (~r (* 100 (hash-ref result 'accuracy)) #:precision '(= 1)))
     (printf "perfect separation: ~a\n" (hash-ref result 'perfect?))]
    [else
     (printf "CUDA not available in this build; skipping.\n")]))

(module+ test
  (require rackunit)

  (when (cuda-available?)
    (define result (run-example))
    (check-equal? (hash-ref result 'prediction-count) 10)
    (check-true   (hash-ref result 'all-valid-probs?))
    (check-true   (hash-ref result 'perfect?))))
