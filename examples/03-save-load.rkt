#lang racket/base

;; Train a regressor, persist it (both to a file and an in-memory bytes
;; blob), reload into a fresh booster, and confirm predictions match
;; bit-for-bit.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/03-save-load.rkt

(require ffi/vector
         racket/file
         racket/format
         xgboost)

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

(define dtrain
  (make-dmatrix features #:nrow 8 #:ncol 3 #:labels labels))

(define booster
  (train dtrain
         #:objective "reg:squarederror"
         #:max-depth 3
         #:eta 0.1
         #:verbosity 0
         #:rounds 50))

(define baseline (predict booster dtrain #:as 'f32vector))

;; ----- File round-trip ---------------------------------------------------

(define model-path (make-temporary-file "xgbrkt-~a.json"))
(save-model booster model-path)
(define file-bytes (file->bytes model-path))
(printf "saved JSON model to ~a (~a bytes)\n"
        (path->string model-path) (bytes-length file-bytes))

(define from-file (load-model model-path))
(define preds-file (predict from-file dtrain #:as 'f32vector))

;; ----- Byte-buffer round-trip (UBJ + JSON) -------------------------------

(define ubj-blob (save-model-to-bytes booster))
(define json-blob (save-model-to-bytes booster #:format "json"))
(printf "ubj  blob: ~a bytes\n" (bytes-length ubj-blob))
(printf "json blob: ~a bytes\n" (bytes-length json-blob))

(define from-ubj (load-model-from-bytes ubj-blob))
(define from-json (load-model-from-bytes json-blob))

(define preds-ubj  (predict from-ubj  dtrain #:as 'f32vector))
(define preds-json (predict from-json dtrain #:as 'f32vector))

;; ----- Verify ------------------------------------------------------------

(define (vec=? a b)
  (and (= (f32vector-length a) (f32vector-length b))
       (for/and ([i (in-range (f32vector-length a))])
         (= (f32vector-ref a i) (f32vector-ref b i)))))

(printf "\n  ~a  matches baseline?\n" (~a "source" #:width 24))
(printf "  ------------------------  -----------------\n")
(for ([row (in-list `(("loaded from file"        ,preds-file)
                      ("loaded from ubj bytes"   ,preds-ubj)
                      ("loaded from json bytes"  ,preds-json)))])
  (printf "  ~a  ~a\n"
          (~a (car row) #:width 24)
          (if (vec=? (cadr row) baseline) "yes" "NO")))

(define (fmt v) (~r v #:precision '(= 4) #:min-width 8))
(printf "\nbaseline predictions:\n")
(for ([i (in-range (f32vector-length baseline))])
  (printf "  i=~a  ~a\n" i (fmt (f32vector-ref baseline i))))

(delete-file model-path)
