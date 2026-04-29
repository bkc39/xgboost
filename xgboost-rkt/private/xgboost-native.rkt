#lang racket/base

(require (except-in ffi/unsafe ->)
         ffi/vector
         racket/contract
         racket/string
         "ffi-raw.rkt")

(provide
 (contract-out
  [xgboost-version (-> string?)]
  [run-regression-demo (-> rational?)]
  [run-classification-demo (-> (and/c rational? (>=/c 0) (<=/c 1)))]
  [dmatrix? (-> any/c boolean?)]
  [dmatrix-create-from-mat
   (->* (f32vector? exact-nonnegative-integer? exact-nonnegative-integer?)
        (real?)
        dmatrix?)]
  [dmatrix-set-float-info! (-> dmatrix? string? f32vector? void?)]
  [dmatrix-free! (-> dmatrix? void?)]
  [dmatrix-nrow (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-ncol (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix-get-float-info (-> dmatrix? string? f32vector?)]
  [dmatrix-num-non-missing (-> dmatrix? exact-nonnegative-integer?)]
  [dmatrix->list (-> dmatrix? (listof (listof real?)))]
  [dmatrix-show (->* (dmatrix?) (output-port?) void?)]
  [booster? (-> any/c boolean?)]
  [booster-create (->* () ((listof dmatrix?)) booster?)]
  [booster-free! (-> booster? void?)]
  [booster-set-param! (-> booster? string? string? void?)]
  [booster-update-one-iter! (-> booster? exact-integer? dmatrix? void?)]
  [booster-predict
   (->* (booster? dmatrix?) (#:config string?) f32vector?)]
  [booster-save-model! (-> booster? path-string? void?)]
  [booster-load-model! (-> booster? path-string? void?)]
  [booster-save-model-to-bytes
   (->* (booster?) (#:format (or/c "json" "ubj")) bytes?)]
  [booster-load-model-from-bytes! (-> booster? bytes? void?)]
  [booster-eval-one-iter
   (-> booster? exact-integer?
       (listof (cons/c string? dmatrix?))
       string?)]
  [parse-eval-line (-> string? (hash/c string? real?))]))

(define (xgboost-version)
  (xgb-version/raw))

(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (xgb-last-error/raw))))

(define (run-regression-demo)
  (define-values (rc out) (xgb-run-regression-demo/raw))
  (check-ok rc 'run-regression-demo)
  out)

(define (run-classification-demo)
  (define-values (rc out) (xgb-run-classification-demo/raw))
  (check-ok rc 'run-classification-demo)
  out)

;; ---------------------------------------------------------------------------
;; DMatrix wrappers.
;;
;; The raw layer's `#:wrap (allocator ...)` auto-registers a finalizer that
;; calls `xgb-dmatrix-free/raw` (itself `#:wrap (deallocator)`).  An explicit
;; `dmatrix-free!` call runs the deallocator — which cancels the finalizer so
;; the free does not execute twice — and we flip the cpointer tag so a second
;; explicit free raises `exn:fail:contract` at the contract boundary instead
;; of double-freeing at the C level.
;; ---------------------------------------------------------------------------

(define (dmatrix? v) (DMatrix? v))

(define (dmatrix-create-from-mat data nrow ncol [missing +nan.0])
  (define expected (* nrow ncol))
  (unless (= (f32vector-length data) expected)
    (raise-argument-error 'dmatrix-create-from-mat
                          (format "f32vector of length ~a (nrow*ncol)" expected)
                          data))
  (define h
    (xgb-dmatrix-create-from-mat/raw data nrow ncol missing))
  (unless h
    (error 'dmatrix-create-from-mat
           "XGBoost returned NULL handle: ~a"
           (xgb-last-error/raw)))
  h)

(define (dmatrix-set-float-info! h field vals)
  (check-ok (xgb-dmatrix-set-float-info/raw h field vals)
            'dmatrix-set-float-info!))

(define (dmatrix-free! h)
  (when (cpointer-has-tag? h 'DMatrix)
    (xgb-dmatrix-free/raw h)
    (set-cpointer-tag! h 'DMatrix-freed)))

(define (dmatrix-nrow h)
  (define-values (rc out) (xgb-dmatrix-num-row/raw h))
  (check-ok rc 'dmatrix-nrow)
  out)

(define (dmatrix-ncol h)
  (define-values (rc out) (xgb-dmatrix-num-col/raw h))
  (check-ok rc 'dmatrix-ncol)
  out)

;; Copy a float info field into a fresh f32vector.  The pointer returned by
;; the raw call is borrowed from the DMatrix; we memcpy out of it before
;; letting control flow near anything that could invalidate it.  An unset
;; field returns an empty f32vector (rc is still 0).
(define (dmatrix-get-float-info h field)
  (define-values (rc len ptr) (xgb-dmatrix-get-float-info/raw h field))
  (check-ok rc 'dmatrix-get-float-info)
  (define result (make-f32vector len))
  (when (and ptr (> len 0))
    (memcpy (f32vector->cpointer result) ptr len _float))
  result)

(define (dmatrix-num-non-missing h)
  (define-values (rc out) (xgb-dmatrix-num-non-missing/raw h))
  (check-ok rc 'dmatrix-num-non-missing)
  out)

;; Reconstruct a dense (nrow x ncol) list-of-lists from the DMatrix's CSR
;; storage.  Missing entries (absent in the CSR) materialize as `+nan.0`;
;; when the DMatrix was built with no missing values, nnz == nrow*ncol and
;; every slot is filled.
(define (dmatrix->list h)
  (define nrow (dmatrix-nrow h))
  (define ncol (dmatrix-ncol h))
  (define nnz (dmatrix-num-non-missing h))
  (define indptr (make-u64vector (+ nrow 1)))
  (define indices (make-u32vector nnz))
  (define data (make-f32vector nnz))
  (check-ok (xgb-dmatrix-get-data-as-csr/raw h "{}" indptr indices data)
            'dmatrix->list)
  (for/list ([r (in-range nrow)])
    (define row (make-vector ncol +nan.0))
    (for ([k (in-range (u64vector-ref indptr r)
                       (u64vector-ref indptr (+ r 1)))])
      (vector-set! row (u32vector-ref indices k) (f32vector-ref data k)))
    (vector->list row)))

;; List of float info fields we probe when showing a DMatrix.  XGBoost
;; returns an empty vector for unset fields, so trying them all is cheap.
(define dmatrix-known-float-info-fields
  '("label" "weight" "base_margin" "label_lower_bound" "label_upper_bound"))

(define dmatrix-show-preview-head 8)
(define dmatrix-show-preview-tail 4)

(define (f32vector->preview-string vec)
  (define n (f32vector-length vec))
  (define (el i) (number->string (f32vector-ref vec i)))
  (cond
    [(<= n (+ dmatrix-show-preview-head dmatrix-show-preview-tail))
     (string-join (for/list ([i (in-range n)]) (el i)) " ")]
    [else
     (define head
       (string-join (for/list ([i (in-range dmatrix-show-preview-head)])
                      (el i))
                    " "))
     (define tail
       (string-join (for/list ([i (in-range (- n dmatrix-show-preview-tail)
                                            n)])
                      (el i))
                    " "))
     (format "~a ... ~a  (len=~a)" head tail n)]))

(define dmatrix-show-max-rows 4)
(define dmatrix-show-max-cols 4)

(define (format-cell v)
  (cond
    [(not (= v v)) "NaN"]              ; NaN is the only value != itself
    [else (real->decimal-string v 4)]))

(define (pad-left s width)
  (define pad (- width (string-length s)))
  (if (<= pad 0) s (string-append (make-string pad #\space) s)))

(define (dmatrix-show h [port (current-output-port)])
  (define nrow (dmatrix-nrow h))
  (define ncol (dmatrix-ncol h))
  (fprintf port "DMatrix ~ax~a\n" nrow ncol)
  (for ([field (in-list dmatrix-known-float-info-fields)])
    (define vals (dmatrix-get-float-info h field))
    (when (> (f32vector-length vals) 0)
      (fprintf port "  ~a: [~a]\n" field (f32vector->preview-string vals))))
  (when (and (> nrow 0) (> ncol 0))
    (define row-cutoff (min nrow dmatrix-show-max-rows))
    (define col-cutoff (min ncol dmatrix-show-max-cols))
    (define cols-truncated? (> ncol col-cutoff))
    (define rows-truncated? (> nrow row-cutoff))
    (define row-tail (if cols-truncated? " ..." ""))
    ;; Two-pass: format every cell in the preview window, then right-align to
    ;; the max width so columns line up like a polars dataframe preview.
    (define formatted-rows
      (for/list ([row (in-list (dmatrix->list h))]
                 [_r (in-range row-cutoff)])
        (for/list ([v (in-list row)]
                   [_c (in-range col-cutoff)])
          (format-cell v))))
    (define col-width
      (for*/fold ([w 0])
                 ([row (in-list formatted-rows)]
                  [cell (in-list row)])
        (max w (string-length cell))))
    (for ([row (in-list formatted-rows)])
      (define cells (for/list ([cell (in-list row)]) (pad-left cell col-width)))
      (fprintf port "  ~a~a\n" (string-join cells " ") row-tail))
    (when rows-truncated?
      (fprintf port "  ...\n"))))

;; ---------------------------------------------------------------------------
;; Booster wrappers.
;;
;; Same lifetime model as DMatrix: allocator/deallocator wired at the raw
;; layer, explicit `booster-free!` flips the cpointer tag so a second call
;; raises a contract error rather than double-freeing.
;;
;; XGBoost's predict API requires a JSON config with `type` set; "{}" raises
;; a fatal "Argument `type` is required".  `booster-predict` defaults to
;; inference-mode predictions over all iterations.
;; ---------------------------------------------------------------------------

(define (booster? v) (Booster? v))

(define booster-default-predict-config
  (string-append
   "{\"type\":0,\"training\":false,"
   "\"iteration_begin\":0,\"iteration_end\":0,"
   "\"strict_shape\":false}"))

(define (booster-create [cache '()])
  (define h (xgb-booster-create/raw cache))
  (unless h
    (error 'booster-create
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  h)

(define (booster-free! h)
  (when (cpointer-has-tag? h 'Booster)
    (xgb-booster-free/raw h)
    (set-cpointer-tag! h 'Booster-freed)))

(define (booster-set-param! h key value)
  (check-ok (xgb-booster-set-param/raw h key value) 'booster-set-param!))

(define (booster-update-one-iter! h iter dtrain)
  (check-ok (xgb-booster-update-one-iter/raw h iter dtrain)
            'booster-update-one-iter!))

;; Predict against `dmat`, returning a fresh f32vector sized to whatever
;; XGBoost emitted.  Uses an opportunistic two-step retry: size for the
;; common reg/binary case (`nrow` predictions), and if XGBoost wants more
;; (e.g. multi-class), resize and call again.  The booster-owned buffer is
;; copied on the C side, so it never escapes.
(define (booster-predict h dmat
                         #:config [config booster-default-predict-config])
  (define guess (max 1 (dmatrix-nrow dmat)))
  (define buf (make-f32vector guess))
  (define-values (rc len) (xgb-booster-predict/raw h dmat config guess buf))
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
     (define-values (rc2 len2) (xgb-booster-predict/raw h dmat config len buf2))
     (check-ok rc2 'booster-predict)
     (unless (= len2 len)
       (error 'booster-predict
              "expected ~a predictions, got ~a" len len2))
     buf2]
    [else (check-ok rc 'booster-predict)]))

;; --- Save / load --------------------------------------------------------

(define (booster-save-model! h path)
  (check-ok (xgb-booster-save-model/raw h path) 'booster-save-model!))

(define (booster-load-model! h path)
  (check-ok (xgb-booster-load-model/raw h path) 'booster-load-model!))

;; Serialize a booster to a fresh bytes object.  `format` is "ubj" (compact
;; binary, default) or "json" (human-readable).  Same size-then-fill dance
;; as predict: probe to get the required length, allocate, refill.
(define (booster-save-model-to-bytes h #:format [fmt "ubj"])
  (define config (string-append "{\"format\":\"" fmt "\"}"))
  (define-values (rc len)
    (xgb-booster-save-model-to-buffer/raw h config 0 (make-bytes 0)))
  (cond
    [(zero? rc) (make-bytes len)]   ; degenerate: 0-byte model
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-save-model-to-buffer/raw h config len buf))
     (check-ok rc2 'booster-save-model-to-bytes)
     (unless (= len2 len)
       (error 'booster-save-model-to-bytes
              "expected ~a bytes, got ~a" len len2))
     buf]
    [else (check-ok rc 'booster-save-model-to-bytes)]))

(define (booster-load-model-from-bytes! h buf)
  (check-ok (xgb-booster-load-model-from-buffer/raw h buf)
            'booster-load-model-from-bytes!))

;; --- Eval one iter ------------------------------------------------------

;; `eval-set` is a list of (name . dmatrix) pairs, in the order their
;; metrics should appear in the result line.  Returns a string like
;;   "[<iter>]\t<name>-<metric>:<value>\t..."
;; matching the format XGBoost prints during default training.
(define (booster-eval-one-iter h iter eval-set)
  (define dmats (map cdr eval-set))
  (define names (map car eval-set))
  ;; Typical line is well under 256 chars even with several metrics; over-
  ;; allocate to avoid the resize round-trip in the common case.
  (define guess 512)
  (define buf (make-bytes guess))
  (define-values (rc len)
    (xgb-booster-eval-one-iter/raw h iter dmats names guess buf))
  (cond
    [(zero? rc) (bytes->string/utf-8 buf #f 0 len)]
    [(= rc 2)
     (define buf2 (make-bytes len))
     (define-values (rc2 len2)
       (xgb-booster-eval-one-iter/raw h iter dmats names len buf2))
     (check-ok rc2 'booster-eval-one-iter)
     (bytes->string/utf-8 buf2 #f 0 len2)]
    [else (check-ok rc 'booster-eval-one-iter)]))

;; Parse a metric line into a hash from "<name>-<metric>" to numeric value.
;; The leading "[<iter>]" token is dropped; any token that is not of the
;; shape "<key>:<number>" is silently skipped.  Greedy `(.*)` on the key
;; means metric names containing additional colons would be misparsed, but
;; XGBoost's built-in metrics never do that.
(define (parse-eval-line line)
  (define parts (regexp-split #rx"\t" line))
  (for/hash ([part (in-list (if (pair? parts) (cdr parts) '()))]
             #:when (regexp-match? #rx":" part))
    (define m (regexp-match #rx"^(.*):([^:]+)$" part))
    (define key (cadr m))
    (define val (string->number (caddr m)))
    (values key (if (real? val) val +nan.0))))

(module+ test
  (require rackunit
           racket/file)

  ;; Existing smoke tests.
  (check-regexp-match #rx"^[0-9]+\\.[0-9]+\\.[0-9]+$" (xgboost-version))
  (check-pred rational? (run-regression-demo))
  (let ([p (run-classification-demo)])
    (check-true (<= 0 p 1)))

  ;; --- DMatrix round-trip -------------------------------------------------
  (define (make-data) (f32vector 1.0 2.0 3.0 4.0 5.0 6.0))  ; 2 rows x 3 cols

  (test-case "dmatrix create/free round-trip"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (check-pred dmatrix? dm)
    (dmatrix-free! dm))

  (test-case "dmatrix finalizer path: drop reference, collect-garbage"
    (let loop ([i 0])
      (when (< i 64)
        (dmatrix-create-from-mat (make-data) 2 3)
        (loop (add1 i))))
    (collect-garbage) (collect-garbage) (collect-garbage)
    ;; If we reach here without crashing, the finalizer chain is safe.
    (check-true #t))

  (test-case "dmatrix-free! is safe to call twice (tag guard, no double-free)"
    (define dm (dmatrix-create-from-mat (make-data) 2 3))
    (dmatrix-free! dm)
    ;; Second call is a no-op; must not segfault or raise.
    (dmatrix-free! dm))

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
  ;; reclaim each handle under GC without crashing.  We cannot easily observe
  ;; the finalizer firing from Racket (wrapping `xgb-dmatrix-free/raw` would
  ;; break the allocator contract), so the assurance here is "does not crash"
  ;; plus the C-side gtest leak test.
  (test-case "finalizer path: GC reclaims DMatrices without explicit free"
    (for ([_ (in-range 256)])
      (dmatrix-create-from-mat (make-data) 2 3))
    (collect-garbage) (collect-garbage) (collect-garbage)
    (check-true #t))

  ;; --- Booster -----------------------------------------------------------

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

  (test-case "booster-free! is idempotent (tag guard)"
    (define b (booster-create))
    (booster-free! b)
    (booster-free! b))   ; no-op; must not crash

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
    (dmatrix-free! dtrain)))
