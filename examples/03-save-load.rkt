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
         "../xgboost-rkt/main.rkt")

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

(define dtrain (dmatrix-create-from-mat features 8 3))
(dmatrix-set-float-info! dtrain "label" labels)

(define booster (booster-create (list dtrain)))
(booster-set-param! booster "objective" "reg:squarederror")
(booster-set-param! booster "max_depth" "3")
(booster-set-param! booster "eta"       "0.1")
(booster-set-param! booster "verbosity" "0")
(for ([iter (in-range 50)])
  (booster-update-one-iter! booster iter dtrain))

(define baseline (booster-predict booster dtrain))

;; ----- File round-trip ---------------------------------------------------

(define model-path (make-temporary-file "xgbrkt-~a.json"))
(booster-save-model! booster model-path)
(define file-bytes (file->bytes model-path))
(printf "saved JSON model to ~a (~a bytes)\n"
        (path->string model-path) (bytes-length file-bytes))

(define from-file (booster-create))
(booster-load-model! from-file model-path)
(define preds-file (booster-predict from-file dtrain))

;; ----- Byte-buffer round-trip (UBJ + JSON) -------------------------------

(define ubj-blob (booster-save-model-to-bytes booster))
(define json-blob (booster-save-model-to-bytes booster #:format "json"))
(printf "ubj  blob: ~a bytes\n" (bytes-length ubj-blob))
(printf "json blob: ~a bytes\n" (bytes-length json-blob))

(define from-ubj (booster-create))
(booster-load-model-from-bytes! from-ubj ubj-blob)

(define from-json (booster-create))
(booster-load-model-from-bytes! from-json json-blob)

(define preds-ubj  (booster-predict from-ubj  dtrain))
(define preds-json (booster-predict from-json dtrain))

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
(booster-free! booster)
(booster-free! from-file)
(booster-free! from-ubj)
(booster-free! from-json)
(dmatrix-free! dtrain)
