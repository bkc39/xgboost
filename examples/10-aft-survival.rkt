#lang racket/base

;; Survival modeling with `survival:aft` on Veterans' lung cancer data.
;;
;; AFT (Accelerated Failure Time) regression handles right-censored
;; survival times — observations where we know the patient was alive
;; at time T but stopped follow-up before death.  The trick is that
;; the "label" isn't a single number; it's an interval:
;;   - exact deaths:  [t, t]   (lower = upper = observed time)
;;   - right-censored: [t, +inf]
;;
;; This is exactly what XGBoost's `label_lower_bound` and
;; `label_upper_bound` info fields are for — *we already expose them*
;; in `dmatrix-set-float-info!`'s whitelist.  No new API was needed
;; for this example; just two info-field calls on the DMatrix.
;;
;; Predictions from `survival:aft` are expected survival times (the
;; mean of the AFT distribution under whichever family we chose).
;; The eval line emits `aft-nloglik` instead of `rmse`.
;;
;; Run from the repo root:
;;   nix develop --command racket examples/10-aft-survival.rkt
;;
;; Source:
;;   Kalbfleisch, J. D. & Prentice, R. L. (1980).
;;   "The Statistical Analysis of Failure Time Data."  Wiley.
;;   CSV mirror: https://vincentarelbundock.github.io/Rdatasets/

(require ffi/vector
         racket/format
         racket/list
         racket/string
         xgboost)

(define veteran-csv #<<VETERAN
rownames,trt,celltype,time,status,karno,diagtime,age,prior
1,1,squamous,72,1,60,7,69,0
2,1,squamous,411,1,70,5,64,10
3,1,squamous,228,1,60,3,38,0
4,1,squamous,126,1,60,9,63,10
5,1,squamous,118,1,70,11,65,10
6,1,squamous,10,1,20,5,49,0
7,1,squamous,82,1,40,10,69,10
8,1,squamous,110,1,80,29,68,0
9,1,squamous,314,1,50,18,43,0
10,1,squamous,100,0,70,6,70,0
11,1,squamous,42,1,60,4,81,0
12,1,squamous,8,1,40,58,63,10
13,1,squamous,144,1,30,4,63,0
14,1,squamous,25,0,80,9,52,10
15,1,squamous,11,1,70,11,48,10
16,1,smallcell,30,1,60,3,61,0
17,1,smallcell,384,1,60,9,42,0
18,1,smallcell,4,1,40,2,35,0
19,1,smallcell,54,1,80,4,63,10
20,1,smallcell,13,1,60,4,56,0
21,1,smallcell,123,0,40,3,55,0
22,1,smallcell,97,0,60,5,67,0
23,1,smallcell,153,1,60,14,63,10
24,1,smallcell,59,1,30,2,65,0
25,1,smallcell,117,1,80,3,46,0
26,1,smallcell,16,1,30,4,53,10
27,1,smallcell,151,1,50,12,69,0
28,1,smallcell,22,1,60,4,68,0
29,1,smallcell,56,1,80,12,43,10
30,1,smallcell,21,1,40,2,55,10
31,1,smallcell,18,1,20,15,42,0
32,1,smallcell,139,1,80,2,64,0
33,1,smallcell,20,1,30,5,65,0
34,1,smallcell,31,1,75,3,65,0
35,1,smallcell,52,1,70,2,55,0
36,1,smallcell,287,1,60,25,66,10
37,1,smallcell,18,1,30,4,60,0
38,1,smallcell,51,1,60,1,67,0
39,1,smallcell,122,1,80,28,53,0
40,1,smallcell,27,1,60,8,62,0
41,1,smallcell,54,1,70,1,67,0
42,1,smallcell,7,1,50,7,72,0
43,1,smallcell,63,1,50,11,48,0
44,1,smallcell,392,1,40,4,68,0
45,1,smallcell,10,1,40,23,67,10
46,1,adeno,8,1,20,19,61,10
47,1,adeno,92,1,70,10,60,0
48,1,adeno,35,1,40,6,62,0
49,1,adeno,117,1,80,2,38,0
50,1,adeno,132,1,80,5,50,0
51,1,adeno,12,1,50,4,63,10
52,1,adeno,162,1,80,5,64,0
53,1,adeno,3,1,30,3,43,0
54,1,adeno,95,1,80,4,34,0
55,1,large,177,1,50,16,66,10
56,1,large,162,1,80,5,62,0
57,1,large,216,1,50,15,52,0
58,1,large,553,1,70,2,47,0
59,1,large,278,1,60,12,63,0
60,1,large,12,1,40,12,68,10
61,1,large,260,1,80,5,45,0
62,1,large,200,1,80,12,41,10
63,1,large,156,1,70,2,66,0
64,1,large,182,0,90,2,62,0
65,1,large,143,1,90,8,60,0
66,1,large,105,1,80,11,66,0
67,1,large,103,1,80,5,38,0
68,1,large,250,1,70,8,53,10
69,1,large,100,1,60,13,37,10
70,2,squamous,999,1,90,12,54,10
71,2,squamous,112,1,80,6,60,0
72,2,squamous,87,0,80,3,48,0
73,2,squamous,231,0,50,8,52,10
74,2,squamous,242,1,50,1,70,0
75,2,squamous,991,1,70,7,50,10
76,2,squamous,111,1,70,3,62,0
77,2,squamous,1,1,20,21,65,10
78,2,squamous,587,1,60,3,58,0
79,2,squamous,389,1,90,2,62,0
80,2,squamous,33,1,30,6,64,0
81,2,squamous,25,1,20,36,63,0
82,2,squamous,357,1,70,13,58,0
83,2,squamous,467,1,90,2,64,0
84,2,squamous,201,1,80,28,52,10
85,2,squamous,1,1,50,7,35,0
86,2,squamous,30,1,70,11,63,0
87,2,squamous,44,1,60,13,70,10
88,2,squamous,283,1,90,2,51,0
89,2,squamous,15,1,50,13,40,10
90,2,smallcell,25,1,30,2,69,0
91,2,smallcell,103,0,70,22,36,10
92,2,smallcell,21,1,20,4,71,0
93,2,smallcell,13,1,30,2,62,0
94,2,smallcell,87,1,60,2,60,0
95,2,smallcell,2,1,40,36,44,10
96,2,smallcell,20,1,30,9,54,10
97,2,smallcell,7,1,20,11,66,0
98,2,smallcell,24,1,60,8,49,0
99,2,smallcell,99,1,70,3,72,0
100,2,smallcell,8,1,80,2,68,0
101,2,smallcell,99,1,85,4,62,0
102,2,smallcell,61,1,70,2,71,0
103,2,smallcell,25,1,70,2,70,0
104,2,smallcell,95,1,70,1,61,0
105,2,smallcell,80,1,50,17,71,0
106,2,smallcell,51,1,30,87,59,10
107,2,smallcell,29,1,40,8,67,0
108,2,adeno,24,1,40,2,60,0
109,2,adeno,18,1,40,5,69,10
110,2,adeno,83,0,99,3,57,0
111,2,adeno,31,1,80,3,39,0
112,2,adeno,51,1,60,5,62,0
113,2,adeno,90,1,60,22,50,10
114,2,adeno,52,1,60,3,43,0
115,2,adeno,73,1,60,3,70,0
116,2,adeno,8,1,50,5,66,0
117,2,adeno,36,1,70,8,61,0
118,2,adeno,48,1,10,4,81,0
119,2,adeno,7,1,40,4,58,0
120,2,adeno,140,1,70,3,63,0
121,2,adeno,186,1,90,3,60,0
122,2,adeno,84,1,80,4,62,10
123,2,adeno,19,1,50,10,42,0
124,2,adeno,45,1,40,3,69,0
125,2,adeno,80,1,40,4,63,0
126,2,large,52,1,60,4,45,0
127,2,large,164,1,70,15,68,10
128,2,large,19,1,30,4,39,10
129,2,large,53,1,60,12,66,0
130,2,large,15,1,30,5,63,0
131,2,large,43,1,60,11,49,10
132,2,large,340,1,80,10,64,10
133,2,large,133,1,75,1,65,0
134,2,large,111,1,60,5,64,0
135,2,large,231,1,70,18,67,10
136,2,large,378,1,80,4,65,0
137,2,large,49,1,30,3,37,0
VETERAN
)

;; ----- Parse + one-hot encode celltype --------------------------------

(define raw-rows
  (for/list ([line (in-list (regexp-split #rx"[\r\n]+" veteran-csv))]
             #:when (positive? (string-length (string-trim line))))
    (string-split line ",")))

(define header (car raw-rows))
(define data-strings (cdr raw-rows))
(printf "loaded ~a rows from embedded CSV\n" (length data-strings))

;; Column indices in the source CSV:
;;   0 rownames | 1 trt | 2 celltype | 3 time | 4 status
;;   5 karno   | 6 diagtime | 7 age   | 8 prior
(define celltype-vocab '("squamous" "smallcell" "adeno" "large"))

(struct ex (features lower upper) #:transparent)

(define (encode-row r)
  (define trt      (string->number (list-ref r 1)))
  (define celltype (list-ref r 2))
  (define time     (string->number (list-ref r 3)))
  (define status   (string->number (list-ref r 4)))
  (define karno    (string->number (list-ref r 5)))
  (define diagtime (string->number (list-ref r 6)))
  (define age      (string->number (list-ref r 7)))
  (define prior    (string->number (list-ref r 8)))
  ;; 4 one-hot columns for celltype.
  (define one-hot
    (for/list ([c (in-list celltype-vocab)])
      (if (equal? c celltype) 1 0)))
  (define features
    (append (list trt karno diagtime age prior) one-hot))
  ;; AFT labels: lower=upper for deaths, upper=+inf for censored.
  (define lower (exact->inexact time))
  (define upper (if (= status 1)
                    (exact->inexact time)
                    +inf.0))
  (ex features lower upper))

(define dataset (map encode-row data-strings))
(define ncol 9)

;; ----- 80/20 split (deterministic, every 5th row → test) --------------

(define-values (train-set test-set)
  (for/fold ([tr '()] [te '()])
            ([row (in-list dataset)] [i (in-naturals)])
    (if (zero? (modulo i 5))
        (values tr (cons row te))
        (values (cons row tr) te))))

(printf "  train: ~a rows, test: ~a rows\n"
        (length train-set) (length test-set))
(printf "  features: trt karno diagtime age prior + 4 celltype one-hots\n")

(define (count-censored xs)
  (for/sum ([x (in-list xs)])
    (if (equal? (ex-upper x) +inf.0) 1 0)))
(printf "  train censoring: ~a/~a right-censored\n"
        (count-censored train-set) (length train-set))

;; ----- Build DMatrices --------------------------------------------------

(define (dataset->dmatrix ds)
  (define n (length ds))
  (define features (make-f32vector (* n ncol)))
  (define lower    (make-f32vector n))
  (define upper    (make-f32vector n))
  (for ([x (in-list ds)] [i (in-naturals)])
    (for ([v (in-list (ex-features x))] [j (in-naturals)])
      (f32vector-set! features (+ (* i ncol) j) (exact->inexact v)))
    (f32vector-set! lower i (ex-lower x))
    (f32vector-set! upper i (ex-upper x)))
  (define dm (make-dmatrix features #:nrow n #:ncol ncol))
  (dmatrix-set-label-lower-bound! dm lower)
  (dmatrix-set-label-upper-bound! dm upper)
  dm)

(define dtrain (dataset->dmatrix train-set))
(define dtest  (dataset->dmatrix test-set))

;; ----- Train -----------------------------------------------------------

(define rounds 100)
(define b
  (train dtrain
         #:evals (list (cons "test" dtest))
         #:objective "survival:aft"
         #:eval-metric "aft-nloglik"
         #:params '(("aft_loss_distribution"       . "normal")
                    ("aft_loss_distribution_scale" . "1.20")
                    ("tree_method"                 . "hist"))
         #:max-depth 3
         #:eta 0.05
         #:verbosity 0
         #:rounds 0))

(define eval-set (list (cons "train" dtrain) (cons "test" dtest)))

(printf "\ntraining (~a rounds, watching aft-nloglik):\n" rounds)
(printf "  ~a  ~a  ~a\n"
        (~a "iter" #:width 4)
        (~a "train-aft-nloglik" #:width 18 #:align 'right)
        (~a "test-aft-nloglik"  #:width 18 #:align 'right))
(for ([iter (in-range rounds)])
  (booster-update-one-iter! b iter dtrain)
  (when (or (< iter 5) (zero? (modulo (+ iter 1) 25)))
    (define m (parse-eval-line (eval-one-iter b iter eval-set)))
    (printf "  ~a  ~a  ~a\n"
            (~a iter #:width 4)
            (~r (hash-ref m "train-aft-nloglik")
                #:precision '(= 4) #:min-width 18)
            (~r (hash-ref m "test-aft-nloglik")
                #:precision '(= 4) #:min-width 18))))

;; ----- Predict + display -----------------------------------------------

(define preds (predict b dtest #:as 'f32vector))
(define test-list test-set)
(define n-test (length test-list))

(define (col s) (~a s #:width 12 #:align 'right))
(define (fmt v) (~r v #:precision '(= 1) #:min-width 12))

(printf "\npredictions on the held-out test set:\n")
(printf "  ~a  ~a  ~a  ~a  ~a\n"
        (~a "i" #:width 3)
        (col "lower")
        (col "upper")
        (col "censored?")
        (col "predicted"))
(for ([x (in-list test-list)] [i (in-range n-test)])
  (define pred (f32vector-ref preds i))
  (define censored? (equal? (ex-upper x) +inf.0))
  (printf "  ~a  ~a  ~a  ~a  ~a\n"
          (~a i #:width 3)
          (fmt (ex-lower x))
          (if censored? (~a "+inf" #:width 12 #:align 'right) (fmt (ex-upper x)))
          (~a (if censored? "yes" "no") #:width 12 #:align 'right)
          (fmt pred)))

;; ----- Sanity checks ---------------------------------------------------

(define n-positive
  (for/sum ([i (in-range n-test)])
    (if (positive? (f32vector-ref preds i)) 1 0)))
(define n-finite
  (for/sum ([i (in-range n-test)])
    (define p (f32vector-ref preds i))
    (if (and (not (equal? p +inf.0)) (= p p)) 1 0)))
(printf "\nsanity:\n")
(printf "  ~a/~a predictions positive (AFT predict = exp(margin))\n"
        n-positive n-test)
(printf "  ~a/~a predictions finite\n" n-finite n-test)

;; For uncensored test rows, summarise residual error in days.
(define uncensored
  (for/list ([x (in-list test-list)] [i (in-range n-test)]
             #:when (not (equal? (ex-upper x) +inf.0)))
    (cons (ex-lower x) (f32vector-ref preds i))))
(when (pair? uncensored)
  (define mae
    (/ (for/sum ([p (in-list uncensored)])
         (abs (- (cdr p) (car p))))
       (length uncensored)))
  (printf "  MAE on ~a uncensored test rows: ~a days\n"
          (length uncensored)
          (~r mae #:precision '(= 1))))
