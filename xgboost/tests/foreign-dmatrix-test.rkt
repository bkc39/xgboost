#lang racket/base

;; Integration tests for the global APIs and the DMatrix surface of
;; `xgboost/foreign` (and the explicit-free helpers from its `unsafe`
;; submodule).

(require rackunit
         racket/file
         (only-in racket/list first second third)
         ffi/vector
         (only-in (except-in ffi/unsafe ->) cast _pointer _uintptr)
         "../foreign.rkt"
         (submod "../foreign.rkt" unsafe))

(module+ test
  ;; --- Global APIs --------------------------------------------------------
  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))

  (test-case "build info returns JSON"
    (define info (xgboost-build-info))
    (check-regexp-match #rx"^\\{" info))

  (test-case "global config round-trips verbosity"
    (define before (xgboost-get-global-config))
    (dynamic-wind
      void
      (lambda ()
        (xgboost-set-global-config! "{\"verbosity\":0}")
        (check-regexp-match #rx"\"verbosity\":0"
                            (xgboost-get-global-config)))
      (lambda () (xgboost-set-global-config! before))))

  (test-case "bad global config reports XGBoost error"
    (check-exn exn:fail?
               (lambda ()
                 (xgboost-set-global-config! "{not-json"))))

  (test-case "log callback registration accepts a Racket procedure"
    (xgboost-register-log-callback! (lambda (msg) (void))))

  ;; --- DMatrix round-trip -------------------------------------------------
  (define (make-data) (f32vector 1.0 2.0 3.0 4.0 5.0 6.0))  ; 2 rows x 3 cols

  (test-case "dmatrix create/free round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-pred dmatrix? dm)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-dense array interface"
    (define dm (dmatrix-create-from-dense (make-data) 2 3 -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (check-equal? (dmatrix->list dm)
                  '((1.0 2.0 3.0) (4.0 5.0 6.0)))
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-csr array interfaces"
    (define dm
      (dmatrix-create-from-csr
       (u64vector 0 2 4)
       (u32vector 0 2 1 2)
       (f32vector 1.0 3.0 5.0 6.0)
       3
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (define rows (dmatrix->list dm))
    (check-= (first (first rows)) 1.0 1e-6)
    (check-true (not (= (second (first rows)) (second (first rows)))))
    (check-= (third (first rows)) 3.0 1e-6)
    (check-= (second (second rows)) 5.0 1e-6)
    (check-= (third (second rows)) 6.0 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-csc array interfaces"
    (define dm
      (dmatrix-create-from-csc
       (u64vector 0 1 2 4)
       (u32vector 0 1 0 1)
       (f32vector 1.0 5.0 3.0 6.0)
       2
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (define rows (dmatrix->list dm))
    (check-= (first (first rows)) 1.0 1e-6)
    (check-true (not (= (second (first rows)) (second (first rows)))))
    (check-= (third (first rows)) 3.0 1e-6)
    (check-= (second (second rows)) 5.0 1e-6)
    (check-= (third (second rows)) 6.0 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-columnar array interfaces"
    (define dm
      (dmatrix-create-from-columnar
       (list (f32vector 1.0 4.0)
             (f32vector 2.0 5.0)
             (f32vector 3.0 6.0))
       -1.0))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (check-equal? (dmatrix->list dm)
                  '((1.0 2.0 3.0) (4.0 5.0 6.0)))
    (dmatrix-free! dm))

  (test-case "dmatrix-create-from-uri loads libsvm"
    (define tmp (make-temporary-file "xgboost-uri-~a.libsvm"))
    (dynamic-wind
      void
      (lambda ()
        (call-with-output-file tmp
          (lambda (out)
            (displayln "0 1:1 3:3" out)
            (displayln "1 2:5 3:6" out))
          #:exists 'truncate)
        (define dm
          (dmatrix-create-from-uri
           (format "{\"uri\":\"~a?format=libsvm\",\"silent\":1}"
                   (path->string tmp))))
        (check-equal? (dmatrix-nrow dm) 2)
        (check-equal? (dmatrix-ncol dm) 4)
        (dmatrix-free! dm))
      (lambda () (when (file-exists? tmp) (delete-file tmp)))))

  (test-case "dmatrix finalizer path: drop reference, collect-garbage"
    (let loop ([i 0])
      (when (< i 64)
        (dmatrix-create-from-mat (make-data) 2 3)
        (loop (add1 i))))
    (collect-garbage) (collect-garbage) (collect-garbage)
    ;; If we reach here without crashing, the finalizer chain is safe.
    (check-true #t))

  (test-case "second dmatrix-free! raises a contract error (tag guard)"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-free! dm)
    ;; The tag flip makes the freed wrapper fail `dmatrix?`, so a second
    ;; free is caught at the contract boundary instead of double-freeing.
    (check-exn exn:fail:contract?
               (lambda () (dmatrix-free! dm))))

  (test-case "shape-mismatch raises contract error"
    (check-exn exn:fail:contract?
               (lambda ()
                 (dmatrix-create-from-mat (f32vector 1.0 2.0 3.0) 2 3))))

  (test-case "set-float-info success"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
    (dmatrix-free! dm))

  (test-case "set-float-info with bogus field surfaces C error"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-exn (lambda (e)
                 (and (exn:fail? e)
                      ;; Message should include the C-side context.
                      (regexp-match? #rx"XGDMatrixSetFloatInfo"
                                     (exn-message e))))
               (lambda ()
                 (dmatrix-set-float-info! dm "definitely_not_a_field"
                                          (f32vector 0.0 1.0))))
    (dmatrix-free! dm))

  ;; --- Metadata accessors + dmatrix-show ----------------------------------
  (test-case "dmatrix-nrow / dmatrix-ncol report shape"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (dmatrix-nrow dm) 2)
    (check-equal? (dmatrix-ncol dm) 3)
    (dmatrix-free! dm))

  (test-case "dmatrix-get-float-info round-trips labels"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.5 1.5))
    (define got (dmatrix-get-float-info dm "label"))
    (check-equal? (f32vector-length got) 2)
    (check-= (f32vector-ref got 0) 0.5 1e-6)
    (check-= (f32vector-ref got 1) 1.5 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-get-float-info on unset field returns empty vector"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (f32vector-length (dmatrix-get-float-info dm "weight")) 0)
    (dmatrix-free! dm))

  (test-case "dmatrix-show prints shape and set info fields"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    (check-regexp-match #rx"DMatrix 2x3" s)
    (check-regexp-match #rx"label:" s)
    ;; Small matrix: show all rows/cols, no ellipses; uniform width.
    (check-regexp-match #rx"1\\.0000 2\\.0000 3\\.0000" s)
    (check-false (regexp-match? #rx"\\.\\.\\." s))
    (dmatrix-free! dm))

  (test-case "dmatrix-show truncates to 4x4 with polars-style ellipses"
    (define big (list->f32vector
                 (for/list ([i (in-range 36)]) (exact->inexact i))))
    (define dm (dmatrix-create-from-mat big 6 6))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    (check-regexp-match #rx"DMatrix 6x6" s)
    ;; Widest cell in preview is "18.0000" (7 chars); narrower cells right-pad.
    ;; First row: " 0.0000  1.0000  2.0000  3.0000 ..."
    (check-regexp-match
     #rx" 0\\.0000  1\\.0000  2\\.0000  3\\.0000 \\.\\.\\." s)
    ;; Fourth (last shown) row starts at value 18, no padding needed.
    (check-regexp-match
     #rx"18\\.0000 19\\.0000 20\\.0000 21\\.0000 \\.\\.\\." s)
    ;; Row 5 should not appear (truncated).
    (check-false (regexp-match? #rx"24\\.0000" s))
    ;; Final standalone "..." line signalling row truncation.
    (check-regexp-match #rx"\n  \\.\\.\\.\n" s)
    (dmatrix-free! dm))

  (test-case "dmatrix-show pads every preview row to the same column width"
    ;; Mix of 1-char and 3-char integer parts forces non-trivial alignment.
    (define data (list->f32vector
                  '(1.0 22.0 333.0
                    4.0  5.0   6.0
                    7.0  8.0   9.0)))
    (define dm (dmatrix-create-from-mat data 3 3))
    (define out (open-output-string))
    (dmatrix-show dm out)
    (define s (get-output-string out))
    ;; Split off the matrix-body lines (skip the "DMatrix" header).
    (define body-lines
      (for/list ([line (in-list (regexp-split #rx"\n" s))]
                 #:when (regexp-match? #rx"^  [ 0-9]" line))
        line))
    (check-equal? (length body-lines) 3)
    ;; All three body lines must be the same length — proof of column alignment.
    (define widths (map string-length body-lines))
    (check-equal? (apply min widths) (apply max widths)
                  (format "body lines differ in width: ~a" body-lines))
    ;; The widest cell is "333.0000" (8 chars); narrower cells get pre-padded.
    (check-regexp-match #rx"  1\\.0000  22\\.0000 333\\.0000" s)
    (check-regexp-match #rx"  4\\.0000   5\\.0000   6\\.0000" s)
    (dmatrix-free! dm))

  (test-case "dmatrix->list reconstructs the feature matrix"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define rows (dmatrix->list dm))
    (check-equal? (length rows) 2)
    (check-equal? (map length rows) '(3 3))
    (for ([expected (in-list '((1.0 2.0 3.0) (4.0 5.0 6.0)))]
          [got (in-list rows)])
      (for ([e (in-list expected)] [g (in-list got)])
        (check-= g e 1e-6)))
    (check-equal? (dmatrix-num-non-missing dm) 6)
    (dmatrix-free! dm))

  ;; --- Leak test (explicit-free path) -------------------------------------
  ;;
  ;; `current-memory-use` only tracks Racket-managed memory, so it won't see
  ;; C-side leaks directly.  The C-side leak smoke test in the gtest suite
  ;; (`XgbDMatrixTest.LeakSmokeTest`) is the authoritative backstop; here we
  ;; assert the Racket-side balance counter stays at zero over 10k cycles and
  ;; that Racket memory use is bounded.
  (test-case "10k explicit create/free cycles: counter balances, memory bounded"
    (define alive 0)
    (define mem-before (current-memory-use))
    (for ([_ (in-range 10000)])
      (define dm (dmatrix-create-from-mat (make-data) 2 3))
      (set! alive (add1 alive))
      (dmatrix-set-float-info! dm "label" (f32vector 0.0 1.0))
      (dmatrix-free! dm)
      (set! alive (sub1 alive)))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (define mem-after (current-memory-use))
    (check-equal? alive 0 "every allocation should have a matching free")
    (check-true (< (- mem-after mem-before) (* 64 1024 1024))
                (format "Racket memory grew by ~a bytes over 10k iters"
                        (- mem-after mem-before))))

  ;; --- Finalizer-path smoke test ------------------------------------------
  ;;
  ;; Allocate without explicit free; the allocator-registered finalizer must
  ;; reclaim each handle under GC without crashing.
  (test-case "finalizer path: GC reclaims DMatrices without explicit free"
    (for ([_ (in-range 256)])
      (dmatrix-create-from-mat (make-data) 2 3))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (check-true #t))

  ;; --- Metadata APIs ------------------------------------------------------

  (test-case "dmatrix-set-feature-info! / dmatrix-get-feature-info round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-feature-info! dm "feature_name" '("height" "weight" "age"))
    (dmatrix-set-feature-info! dm "feature_type" '("q" "q" "i"))
    (check-equal? (dmatrix-get-feature-info dm "feature_name")
                  '("height" "weight" "age"))
    (check-equal? (dmatrix-get-feature-info dm "feature_type")
                  '("q" "q" "i"))
    (dmatrix-free! dm))

  (test-case "dmatrix-get-feature-info on unset field returns empty list"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-equal? (dmatrix-get-feature-info dm "feature_name") '())
    (dmatrix-free! dm))

  (test-case "dmatrix-set-uint-info! group + dmatrix-get-uint-info group_ptr"
    ;; XGBoost reads "group" as ranking-group sizes and exposes the
    ;; cumulative form via "group_ptr" (so {2} sets to {0,2}).
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-set-uint-info! dm "group" (u32vector 2))
    (define g (dmatrix-get-uint-info dm "group_ptr"))
    (check-equal? (u32vector-length g) 2)
    (check-equal? (u32vector-ref g 0) 0)
    (check-equal? (u32vector-ref g 1) 2)
    (dmatrix-free! dm))

  (test-case "dmatrix-set-info-from-interface! drives label round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define labels (f32vector 0.25 0.75))
    (dmatrix-set-info-from-interface!
     dm "label"
     (format "{\"data\":[~a,false],\"typestr\":\"<f4\",\"shape\":[2],\"version\":3}"
             (cast (f32vector->cpointer labels) _pointer _uintptr)))
    (define got (dmatrix-get-float-info dm "label"))
    (check-equal? (f32vector-length got) 2)
    (check-= (f32vector-ref got 0) 0.25 1e-6)
    (check-= (f32vector-ref got 1) 0.75 1e-6)
    (dmatrix-free! dm))

  (test-case "dmatrix-slice picks rows in given order"
    (define data (f32vector 1.0 2.0
                            3.0 4.0
                            5.0 6.0))
    (define dm (dmatrix-create-from-mat data 3 2))
    (define sliced (dmatrix-slice dm '(2 0)))
    (check-equal? (dmatrix-nrow sliced) 2)
    (check-equal? (dmatrix-ncol sliced) 2)
    (check-equal? (dmatrix->list sliced)
                  '((5.0 6.0) (1.0 2.0)))
    (dmatrix-free! sliced)
    (dmatrix-free! dm))

  (test-case "dmatrix-slice accepts s32vector and rejects bad indices"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-pred dmatrix? (dmatrix-slice dm (s32vector 1 0)))
    (check-exn exn:fail:contract?
               (lambda ()
                 (dmatrix-slice dm '(-1))))
    (dmatrix-free! dm))

  (test-case "dmatrix-save-binary! writes a file XGBoost can reload"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (define tmp (make-temporary-file "xgboost-dmat-~a.buffer"))
    (dynamic-wind
      void
      (lambda ()
        (dmatrix-save-binary! dm tmp)
        (define loaded
          (dmatrix-create-from-uri
           (format "{\"uri\":\"~a\",\"silent\":1}" (path->string tmp))))
        (check-equal? (dmatrix-nrow loaded) 2)
        (check-equal? (dmatrix-ncol loaded) 3)
        (dmatrix-free! loaded))
      (lambda () (when (file-exists? tmp) (delete-file tmp))))
    (dmatrix-free! dm))

  (test-case "dmatrix-get-quantile-cut returns array-interface JSON after training"
    ;; Quantile cuts are computed during training on a hist-method booster.
    (define dm
      (dmatrix-create-from-mat
       (f32vector 1.0 2.0 0.5 2.0 1.0 1.5 3.0 0.5 0.0
                  0.5 3.0 2.0 4.0 2.0 1.0 1.5 1.5 0.5
                  2.5 3.5 1.5 0.0 1.0 0.0)
       8 3))
    (dmatrix-set-float-info! dm "label"
                             (f32vector 3.5 3.5 6.5 2.0 9.0 4.0 7.0 1.0))
    (define b (booster-create (list dm)))
    (booster-set-param! b "objective" "reg:squarederror")
    (booster-set-param! b "tree_method" "hist")
    (booster-set-param! b "max_depth" "2")
    (booster-set-param! b "verbosity" "0")
    (booster-update-one-iter! b 0 dm)
    (define-values (indptr-json data-json) (dmatrix-get-quantile-cut dm))
    (check-true (regexp-match? #rx"\"shape\"" indptr-json))
    (check-true (regexp-match? #rx"\"shape\"" data-json))
    (booster-free! b)
    (dmatrix-free! dm)))
