#lang racket/base

;; Booster inspection: attributes, string feature info, model dumps, feature
;; importance scores, and per-iteration evaluation lines.

(require ffi/vector
         racket/string
         "../error.rkt"
         "../raw/booster.rkt")

(provide booster-set-attr!
         booster-delete-attr!
         booster-get-attr
         booster-get-attr-names
         booster-set-feature-info!
         booster-get-feature-info
         booster-dump-model
         booster-dump-model-with-features
         booster-feature-score
         booster-eval-one-iter
         parse-eval-line)

(define (booster-set-attr! h key value)
  (check-ok (xgb-booster-set-attr/raw h key value)
            'booster-set-attr!))

(define (booster-delete-attr! h key)
  (check-ok (xgb-booster-delete-attr/raw h key)
            'booster-delete-attr!))

(define (booster-get-attr h key)
  (define-values (rc len found)
    (xgb-booster-get-attr/raw h key 0 (make-bytes 0)))
  (cond
    [(and (zero? rc) (zero? found)) #f]
    [(zero? rc) ""]
    [(= rc 2)
     (define buf (make-bytes len))
     (define-values (rc2 len2 found2)
       (xgb-booster-get-attr/raw h key len buf))
     (check-ok rc2 'booster-get-attr)
     (cond
       [(zero? found2) #f]
       [else (bytes->string/utf-8 buf #f 0 len2)])]
    [else (check-ok rc 'booster-get-attr)]))

(define (booster-get-attr-names h)
  (copy-nul-separated-result
   'booster-get-attr-names
   (lambda (capacity buf)
     (xgb-booster-get-attr-names/raw h capacity buf))))

(define (booster-set-feature-info! h field vals)
  (check-ok (xgb-booster-set-str-feature-info/raw h field vals)
            'booster-set-feature-info!))

(define (booster-get-feature-info h field)
  (copy-nul-separated-result
   'booster-get-feature-info
   (lambda (capacity buf)
     (xgb-booster-get-str-feature-info/raw h field capacity buf))))

(define (booster-dump-model h #:format [fmt "text"] #:with-stats? [stats? #f])
  (copy-nul-separated-result
   'booster-dump-model
   (lambda (capacity buf)
     (xgb-booster-dump-model/raw h fmt (if stats? 1 0) capacity buf))))

(define (booster-dump-model-with-features h feature-names feature-types
                                          #:format [fmt "text"]
                                          #:with-stats? [stats? #f])
  (unless (= (length feature-names) (length feature-types))
    (error 'booster-dump-model-with-features
           "feature name count ~a does not match feature type count ~a"
           (length feature-names)
           (length feature-types)))
  (copy-nul-separated-result
   'booster-dump-model-with-features
   (lambda (capacity buf)
     (xgb-booster-dump-model-with-features/raw h feature-names feature-types
                                               fmt (if stats? 1 0)
                                               capacity buf))))

(define (json-quote-string s)
  (format "~s" s))

(define (feature-score-config importance-type feature-names)
  (define base (format "\"importance_type\":~a"
                       (json-quote-string importance-type)))
  (define names
    (if feature-names
        (format ",\"feature_names\":[~a]"
                (string-join (map json-quote-string feature-names) ","))
        ""))
  (format "{~a~a}" base names))

(define (u64vector->list vec)
  (for/list ([i (in-range (u64vector-length vec))])
    (u64vector-ref vec i)))

(define (booster-feature-score h
                               #:importance-type [importance-type "weight"]
                               #:feature-names [feature-names #f]
                               #:config [config #f])
  (define cfg (or config (feature-score-config importance-type feature-names)))
  (define-values (rc feature-len n-features dim n-scores)
    (xgb-booster-feature-score/raw h cfg
                                   0 (make-bytes 0)
                                   0 (make-u64vector 0)
                                   0 (make-f32vector 0)))
  (cond
    [(zero? rc)
     (hash 'features '()
           'shape '()
           'scores (make-f32vector 0))]
    [(= rc 2)
     (define feature-buf (make-bytes feature-len))
     (define shape-buf (make-u64vector dim))
     (define score-buf (make-f32vector n-scores))
     (define-values (rc2 feature-len2 n-features2 dim2 n-scores2)
       (xgb-booster-feature-score/raw h cfg
                                      feature-len feature-buf
                                      dim shape-buf
                                      n-scores score-buf))
     (check-ok rc2 'booster-feature-score)
     (unless (and (= feature-len2 feature-len)
                  (= n-features2 n-features)
                  (= dim2 dim)
                  (= n-scores2 n-scores))
       (error 'booster-feature-score
              "feature score output shape changed during copy"))
     (hash 'features (nul-separated-bytes->strings feature-buf n-features2)
           'shape (u64vector->list shape-buf)
           'scores score-buf)]
    [else (check-ok rc 'booster-feature-score)]))

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
