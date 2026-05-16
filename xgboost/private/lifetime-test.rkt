#lang racket/base

(module+ test
  (require ffi/vector
           racket/list
           rackunit
           (prefix-in ffi: "../foreign.rkt")
           "../main.rkt")

  (define (force-gc!)
    (for ([_ (in-range 3)])
      (collect-garbage)))

  (test-case "DMatrix slice survives parent GC"
    (define slice
      (let ()
        (define parent
          (ffi:dmatrix-create-from-dense
           (f32vector 1.0 2.0
                      3.0 4.0
                      5.0 6.0)
           3 2))
        (ffi:dmatrix-slice parent '(2 0))))
    (force-gc!)
    (check-equal? (ffi:dmatrix->list slice)
                  '((5.0 6.0) (1.0 2.0))))

  (test-case "Booster retains training DMatrix across GC"
    (define b
      (let ()
        (define dtrain
          (make-dmatrix '((1.0 2.0) (2.0 3.0) (3.0 4.0) (4.0 5.0))
                        #:labels '(1.0 2.0 3.0 4.0)))
        (train dtrain
               #:objective "reg:squarederror"
               #:verbosity 0
               #:rounds 5)))
    (force-gc!)
    (define cached-dm (first (booster-cache b)))
    (define line (eval-one-iter b 4 (list (cons "train" cached-dm))))
    (check-true (hash-has-key? (parse-eval-line line) "train-rmse")))

  (test-case "Low-level booster-create retains cache DMatrices across GC"
    ;; ffi:booster-create receives a list of DMatrix cpointers and returns a
    ;; bare booster cpointer.  The wrapper must pin the cache list to the
    ;; booster's lifetime; without that, dropping the user-side DMatrix
    ;; reference would make the booster's update_one_iter touch freed memory.
    ;; Drop the local DMatrix reference inside the let; only the booster's
    ;; internal cache table holds it after the let returns.
    (define b
      (let ()
        (define dm
          (ffi:dmatrix-create-from-dense
           (f32vector 1.0 2.0
                      3.0 4.0
                      5.0 6.0
                      7.0 8.0)
           4 2))
        (ffi:dmatrix-set-float-info! dm "label" (f32vector 1.0 2.0 3.0 4.0))
        (define booster (ffi:booster-create (list dm)))
        (ffi:booster-set-param! booster "objective" "reg:squarederror")
        (ffi:booster-set-param! booster "verbosity" "0")
        (ffi:booster-update-one-iter! booster 0 dm)
        booster))
    (force-gc!)
    ;; Predict on a fresh DMatrix.  If the cache retention failed, the
    ;; booster's internal cache slot points at freed C-side memory, which
    ;; XGBoost may probe during inference.  Either way, the booster must
    ;; still produce predictions of the right shape.
    (define dm2
      (ffi:dmatrix-create-from-dense
       (f32vector 0.5 0.5
                  1.0 1.0)
       2 2))
    (define preds (ffi:booster-predict b dm2))
    (check-equal? (f32vector-length preds) 2))

  (test-case "Repeated train/predict cycles do not leak Racket-heap memory"
    ;; current-memory-use measures Racket-managed memory only; XGBoost's
    ;; native allocations live off-heap and won't show here.  This test
    ;; catches catastrophic wrapper-struct accumulation, not C-side leaks.
    (define (one-cycle)
      (define dm
        (make-dmatrix '((1.0 2.0 0.5)
                        (2.0 1.0 1.5)
                        (3.0 0.5 0.0)
                        (0.5 3.0 2.0))
                      #:labels '(3.5 3.5 6.5 2.0)))
      (define b (train dm
                       #:objective "reg:squarederror"
                       #:max-depth 2
                       #:eta 0.2
                       #:verbosity 0
                       #:rounds 5))
      (predict b dm))
    (one-cycle)
    (force-gc!)
    (define baseline (current-memory-use))
    (for ([_ (in-range 200)])
      (one-cycle))
    (force-gc!)
    (define growth (- (current-memory-use) baseline))
    (check-true (< growth (* 200 1024 1024))
                (format "Racket heap grew by ~a bytes after 200 cycles" growth))))
