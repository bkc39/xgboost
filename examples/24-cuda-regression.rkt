#lang racket/base

;; CUDA-accelerated regression training.
;;
;; Trains the same synthetic dataset as examples/01-train-regression.rkt but
;; with device=cuda and tree_method=hist, delegating tree construction to the GPU.
;;
;; Requires a CUDA-enabled libxgbcompat (built with ./scripts/build-so.sh linux-cuda
;; or nix build .#cpp-cuda) and a physical NVIDIA GPU at runtime.
;;
;; Run from the repo root:
;;   nix develop .#cuda --command racket examples/24-cuda-regression.rkt

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
  (f32vector 1.0 2.0 0.5
             2.0 1.0 1.5
             3.0 0.5 0.0
             0.5 3.0 2.0
             4.0 2.0 1.0
             1.5 1.5 0.5
             2.5 3.5 1.5
             0.0 1.0 0.0))

(define labels (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))

(define (run-example)
  (define dtrain (dmatrix-create-from-mat features 8 3))
  (dmatrix-set-float-info! dtrain "label" labels)

  (define booster (booster-create (list dtrain)))
  (booster-set-param! booster "device"      "cuda")
  (booster-set-param! booster "tree_method" "hist")
  (booster-set-param! booster "objective"   "reg:squarederror")
  (booster-set-param! booster "max_depth"   "3")
  (booster-set-param! booster "eta"         "0.1")
  (booster-set-param! booster "verbosity"   "0")

  (define rounds 50)
  (for ([iter (in-range rounds)])
    (booster-update-one-iter! booster iter dtrain))

  (define preds (booster-predict booster dtrain))
  (define n (f32vector-length labels))
  (define mse
    (/ (for/sum ([i (in-range n)])
         (define d (- (f32vector-ref preds i) (f32vector-ref labels i)))
         (* d d))
       n))

  (define result
    (hash 'prediction-count n
          'mse              mse
          'improved?        (< mse 1.0)))

  (booster-free! booster)
  (dmatrix-free! dtrain)
  result)

(module+ main
  (cond
    [(cuda-available?)
     (define result (run-example))
     (printf "CUDA regression predictions: ~a\n"
             (hash-ref result 'prediction-count))
     (printf "training MSE: ~a\n"
             (~r (hash-ref result 'mse) #:precision '(= 6)))
     (printf "MSE < 1.0: ~a\n" (hash-ref result 'improved?))]
    [else
     (printf "CUDA not available in this build; skipping.\n")]))

(module+ test
  (require rackunit)

  (when (cuda-available?)
    (define result (run-example))
    (check-equal? (hash-ref result 'prediction-count) 8)
    (check-true   (hash-ref result 'improved?))
    (check-true   (< (hash-ref result 'mse) 1.0))))
