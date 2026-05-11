#lang racket/base

(provide pre-installer)

(require racket/file)

(define (copy-native-libs! dest-dir source-dir pattern)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))])
    (when (regexp-match? pattern (path->string f))
      (define src (build-path source-dir f))
      (define dst (build-path dest-dir f))
      (when (file-exists? dst)
        (delete-file dst))
      (copy-file src dst))))

(define (has-matching-files? dir pattern)
  (and (directory-exists? dir)
       (pair? (filter (lambda (f) (regexp-match? pattern (path->string f)))
                      (directory-list dir)))))

; On x86_64 Linux prefer a CUDA build if staged, then fall back to CPU.
; On aarch64 Linux use the dedicated cross-compiled candidate.
(define (candidate-dirs this-collection-path)
  (define base (build-path this-collection-path "native-libs" "candidates"))
  (define platform-dirs
    (case (system-type 'os)
      [(macosx) '("darwin")]
      [(unix)
       (if (regexp-match? #rx"aarch64" (path->string (system-library-subpath #f)))
           '("linux-aarch64")
           '("linux-cuda" "linux-cpu"))]
      [else '()]))
  (map (lambda (d) (build-path base d)) platform-dirs))

(define (pre-installer collections-top-path this-collection-path user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define cpp-lib-path (getenv "XGBOOST_NATIVE_LIB_PATH"))
  ; Matches libxgbcompat, libxgboost, libgomp, and libstdc++ so all bundled
  ; libs are copied from candidates alongside libxgbcompat. libstdc++ is
  ; bundled on Linux so libxgboost.so resolves GLIBCXX symbols via RPATH=$ORIGIN
  ; on hosts whose system libstdc++ is too old (e.g. pkg-build.racket-lang.org).
  (define pattern #rx"^lib(xgbcompat|xgboost|gomp|omp|stdc\\+\\+)\\.")
  (define installed-pattern #rx"^libxgbcompat\\.")
  (cond
    [cpp-lib-path
     (copy-native-libs! native-libs-dir (build-path cpp-lib-path "lib") pattern)]
    [(has-matching-files? native-libs-dir installed-pattern)
     (void)]
    [else
     (define candidate
       (for/first ([d (in-list (candidate-dirs this-collection-path))]
                   #:when (has-matching-files? d installed-pattern))
         d))
     (if candidate
         (copy-native-libs! native-libs-dir candidate pattern)
         (error 'pre-installer
                "xgbcompat library not found. Either:\n  1. Run: ./scripts/build-so.sh linux|darwin, then raco pkg install\n  2. Set XGBOOST_NATIVE_LIB_PATH to a dir containing lib/libxgbcompat.*\n  3. Copy libxgbcompat.* manually to ~a"
                (path->string native-libs-dir)))]))
