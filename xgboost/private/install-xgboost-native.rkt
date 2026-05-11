#lang racket/base

(provide pre-installer)

(require racket/file
         file/gunzip
         file/untar)

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

;; Extract <candidate>.tar.gz into dest-dir.  The tarball has a single
;; top-level directory (the candidate name); strip-count 1 places its
;; contents directly into dest-dir.  Uses Racket's built-in gunzip/untar
;; so the install does not depend on system `tar` or `gzip` binaries.
(define (extract-tgz-to! tgz-path dest-dir)
  (make-directory* dest-dir)
  (define-values (gz-in gz-out) (make-pipe (* 4 1024 1024)))
  (define gunzip-thread
    (thread
     (lambda ()
       (call-with-input-file tgz-path
         (lambda (in) (gunzip-through-ports in gz-out)))
       (close-output-port gz-out))))
  (untar gz-in #:dest dest-dir #:strip-count 1)
  (thread-wait gunzip-thread))

; On x86_64 Linux prefer a CUDA build if staged, then fall back to CPU.
; On aarch64 Linux use the dedicated cross-compiled candidate.
(define (candidate-names)
  (case (system-type 'os)
    [(macosx) '("darwin")]
    [(unix)
     (if (regexp-match? #rx"aarch64" (path->string (system-library-subpath #f)))
         '("linux-aarch64")
         '("linux-cuda" "linux-cpu"))]
    [else '()]))

(define (pre-installer collections-top-path this-collection-path user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define candidates-base (build-path native-libs-dir "candidates"))
  (define cpp-lib-path (getenv "XGBOOST_NATIVE_LIB_PATH"))
  ; Matches every bundled .so/.dylib copied alongside libxgbcompat:
  ;   libxgbcompat - the C++ wrapper Racket FFI binds to
  ;   libxgboost   - the upstream XGBoost runtime
  ;   libgomp/libomp - OpenMP runtime used by libxgboost
  ;   libstdc++    - bundled on Linux so libxgboost finds the GLIBCXX symbols
  ;                  it was built against on hosts with an older system one
  ;   libxgbshim   - tiny Linux glibc-compat shim (see scripts/glibc-shim.c)
  ;                  that lets polyfill-glibc rewrite the libs to a glibc 2.17
  ;                  baseline so pkg-build.racket-lang.org can load them
  (define pattern #rx"^lib(xgbcompat|xgboost|gomp|omp|stdc\\+\\+|xgbshim)\\.")
  (define installed-pattern #rx"^libxgbcompat\\.")
  (cond
    [cpp-lib-path
     (copy-native-libs! native-libs-dir (build-path cpp-lib-path "lib") pattern)]
    [(has-matching-files? native-libs-dir installed-pattern)
     (void)]
    [else
     ; For each candidate in priority order:
     ;   - If the unpacked dir exists, copy matching files (fast path).
     ;   - Else if <candidate>.tar.gz exists, extract it into the unpacked
     ;     dir first, then copy.  This lets us ship oversized candidates
     ;     (e.g. CUDA libxgboost.so > 100 MB) as a gzip tarball.
     (define chosen
       (for/first ([name (in-list (candidate-names))]
                   #:when
                   (let ([dir (build-path candidates-base name)]
                         [tgz (build-path candidates-base (string-append name ".tar.gz"))])
                     (or (has-matching-files? dir installed-pattern)
                         (file-exists? tgz))))
         name))
     (cond
       [(not chosen)
        (error 'pre-installer
               "xgbcompat library not found. Either:\n  1. Run: ./scripts/build-so.sh linux|darwin, then raco pkg install\n  2. Set XGBOOST_NATIVE_LIB_PATH to a dir containing lib/libxgbcompat.*\n  3. Copy libxgbcompat.* manually to ~a"
               (path->string native-libs-dir))]
       [else
        (define dir (build-path candidates-base chosen))
        (unless (has-matching-files? dir installed-pattern)
          (extract-tgz-to!
           (build-path candidates-base (string-append chosen ".tar.gz"))
           dir))
        (copy-native-libs! native-libs-dir dir pattern)])]))
