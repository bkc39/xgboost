---
name: racket-ffi
description: |
  Reference patterns for wrapping a C/C++ library as a Racket FFI binding when
  the library owns heap resources (paired create/free). Use this when adding
  a new opaque handle type, designing an alloc/free wrapper, or debugging
  finalizer / leak issues.
---

# Racket FFI: wrapping C libraries that own resources

Primary reference: <https://docs.racket-lang.org/foreign/intro.html>. Pay
particular attention to the "Reliable Release of Resources" section, which
establishes the canonical allocator/deallocator pattern reproduced below.

This skill codifies the template this repo uses for every opaque handle —
`DMatrix` in Phase 1, `Booster` next. When in doubt, replicate the DMatrix
wiring end-to-end (`ffi-raw.rkt` → `xgboost-native.rkt` → `main.rkt` → tests).

---

## 1. When to apply

Use this pattern when the C API:

- Exposes an opaque handle (typedef'd pointer) that must be freed by a
  paired function.
- Could leak if an exception escapes mid-construction.
- Returns errors via rc + a thread-local message buffer, *or* returns
  `NULL` on failure.

Do not use it for:

- Pure functions with no C-owned state (bind directly, no finalizer).
- Buffers owned by a parent handle (e.g., a prediction array returned by
  a Booster — lifetime is the Booster's, so copy into Racket-owned storage
  before the parent goes away).

---

## 2. Opaque handle type

```racket
(require ffi/unsafe)

(define-cpointer-type _DMatrix)
;; Generates three bindings:
;;   _DMatrix        — ctype for non-null pointers; FFI errors on NULL input
;;   _DMatrix/null   — ctype that accepts/produces NULL (used for NULL-on-error)
;;   DMatrix?        — predicate: any cpointer tagged 'DMatrix
```

Convention: underscore prefix is a C type; the bare name (`DMatrix?`) is the
Racket predicate. The tag is a symbol interned from the type name; all
pointers flowing through `_DMatrix` carry that tag.

---

## 3. Allocator / deallocator template

From `ffi/unsafe/alloc`. **Define the deallocator first** — the allocator
references it.

```racket
(require ffi/unsafe ffi/unsafe/alloc ffi/unsafe/define ffi/vector)

(define-ffi-definer define-xgb (ffi-lib "libxgbcompat"))

(define-cpointer-type _DMatrix)

(define-xgb xgb-dmatrix-free/raw
  (_fun _DMatrix -> _void)
  #:c-id xgb_dmatrix_free
  #:wrap (deallocator))

(define-xgb xgb-dmatrix-create-from-mat/raw
  (_fun (data : _f32vector) (nrow : _size) (ncol : _size) (missing : _float)
        -> _DMatrix/null)                 ; NULL = error (see §4)
  #:c-id xgb_dmatrix_create_from_mat
  #:wrap (allocator xgb-dmatrix-free/raw))
```

Semantics:

- `allocator` wraps a constructor; on each non-`#f` return it registers a
  finalizer that calls the paired `deallocator`.
- `deallocator` cancels the finalizer for its argument when called explicitly,
  so explicit `free` followed by GC is never a double free.
- Allocator and deallocator both run in atomic mode. Do **not** call back
  into Racket-level mutable state from them.

---

## 4. NULL-on-error vs rc+out-pointer

Two idioms for reporting failure in the `extern "C"` layer; pick per API.

**NULL-on-error** (used for handle constructors, because it matches the
`#:wrap (allocator ...)` contract directly):

```c
xgb_dmatrix_t xgb_dmatrix_create_from_mat(const float*, size_t, size_t, float);
// Returns NULL on failure. Caller reads xgb_last_error() for the message.
```

```racket
(define (dmatrix-create-from-mat data nrow ncol [missing +nan.0])
  (define h (xgb-dmatrix-create-from-mat/raw data nrow ncol missing))
  (unless h
    (error 'dmatrix-create-from-mat
           "XGBoost returned NULL handle: ~a" (xgb-last-error/raw)))
  h)
```

**rc + out-pointer** (used for mutating methods on an existing handle):

```c
int xgb_dmatrix_set_float_info(xgb_dmatrix_t h, const char* f, const float* v, size_t n);
// 0 = success, non-zero = error.
```

```racket
(define (check-ok rc who)
  (unless (zero? rc)
    (error who "FFI call failed (rc=~a): ~a" rc (xgb-last-error/raw))))
```

Both rely on the same thread-local `xgb_last_error()` buffer. The C layer's
`check()` helper populates it on any non-zero rc from the underlying XGBoost
API.

---

## 5. Passing arrays

Use SRFI-4 vectors from `ffi/vector`:

- `_f32vector` — contiguous `float*`, malloc-backed, non-moving under GC.
- `_f64vector` — same for `double*`.
- `_s32vector`, `_u64vector`, etc.

```racket
(define-xgb xgb-dmatrix-set-float-info/raw
  (_fun _DMatrix _string/utf-8
        (vals : _f32vector) (_size = (f32vector-length vals))
        -> _int)
  #:c-id xgb_dmatrix_set_float_info)
```

The `(_size = (f32vector-length vals))` form auto-derives the length from an
earlier parameter — the caller never passes it.

Avoid `(_list i _float)` / `(_list i _double)` in hot paths: it copies each
element into a fresh malloc buffer on every call. Fine for short
`SetParam`-style arrays.

---

## 6. Double-free guard

The deallocator cancels the allocator-registered finalizer, so a single
explicit free is correct. But a *second* explicit free would try to free a
dangling handle and segfault. Flip the cpointer tag to make that cheap to
detect:

```racket
(define (dmatrix-free! h)
  (when (cpointer-has-tag? h 'DMatrix)
    (xgb-dmatrix-free/raw h)
    (set-cpointer-tag! h 'DMatrix-freed)))
```

After the flip, the contracted entry point (`(-> dmatrix? ...)`) rejects the
pointer before it reaches the raw free. This matches the explicit-close
idiom used by `ffi/unsafe/port`.

---

## 7. Name collisions: `->`

`racket/contract` and `ffi/unsafe` both export `->`. If a wrapper file needs
both, use `(except-in ffi/unsafe ->)`:

```racket
(require (except-in ffi/unsafe ->)
         ffi/vector
         racket/contract)
```

Alternatively, `require` only the specific names you need from `ffi/unsafe`
(`cpointer-has-tag?`, `set-cpointer-tag!`, …) and get vectors from
`ffi/vector`.

---

## 8. `real->single-flonum` and `_float`

`real->single-flonum` is unsupported on Racket CS (the default). **Do not
pre-convert** reals before passing them through `_float`: the FFI ctype
handles the coercion itself. The one exception is NaN sentinel values —
`+nan.0` passes through `_float` cleanly.

---

## 9. Finalizer gotchas

- Finalizers run on a dedicated thread; they must be safe w.r.t. any thread
  that might hold the library. Every XGBoost `*Free` entry point is
  re-entrant and safe for distinct handles — verify the equivalent for new
  libraries before adding a binding.
- Never mutate Racket-level state inside a deallocator wrapper — the
  allocator/deallocator combinators run in atomic mode.
- Racket does not `dlclose` an `ffi-lib` at process exit by default, so the
  "finalizer fires after library unload" hazard is moot for long-lived
  processes. Do not rely on this if the library is loaded dynamically by
  something that *does* unload it.

---

## 10. Leak tests

Two layers, complementary:

**C-side gtest** — authoritative; measures RSS over a tight
create/use/free loop using `getrusage(RUSAGE_SELF, …)`. Units differ
(`ru_maxrss` = bytes on macOS, KB on Linux — branch on `__APPLE__`). A
leaking implementation grows RSS by hundreds of MB; 50 MB is a generous
smoke threshold.

**Racket-side** — `current-memory-use` only tracks Racket-managed memory, so
it **will not** see C-side leaks. Use a balance counter instead:

```racket
(define alive 0)
(for ([_ (in-range 10000)])
  (define dm (dmatrix-create-from-mat (make-data) 2 3))
  (set! alive (add1 alive))
  (dmatrix-free! dm)
  (set! alive (sub1 alive)))
(collect-garbage) (collect-garbage) (collect-garbage)
(check-equal? alive 0))
```

For the finalizer path, a "does not crash" loop + GC sweep is the best
Racket can do without instrumentation:

```racket
(for ([_ (in-range 256)])
  (dmatrix-create-from-mat (make-data) 2 3))   ; drop reference
(collect-garbage) (collect-garbage) (collect-garbage)
```

Real leak hunting for finalizer paths belongs in the C gtest (valgrind/ASan
are future work).

---

## 11. Links

- <https://docs.racket-lang.org/foreign/intro.html> — Racket FFI guide.
- <https://docs.racket-lang.org/foreign/intro.html#%28part._.Reliable_.Release_of_.Resources%29>
  — the allocator/deallocator pattern.
- <https://docs.racket-lang.org/foreign/Allocation_and_Finalization.html>
  — `ffi/unsafe/alloc` reference (`allocator`, `deallocator`, `releaser`).
- <https://docs.racket-lang.org/foreign/Pointer_Types.html> — `_cpointer`,
  `define-cpointer-type`, `cpointer-has-tag?`, `set-cpointer-tag!`.
- <https://docs.racket-lang.org/foreign/homogeneous-vectors.html> —
  `_f32vector` / `_f64vector` / etc. from `ffi/vector`.

---

## Applying this in `xgboost-rkt`

Every new XGBoost handle type (Booster next, then anything with a
`*Handle`) should mirror the DMatrix wiring:

1. `extern "C"` layer in `cpp/include/xgbcompat/xgbcompat.hpp` +
   `cpp/src/xgbcompat.cpp`: constructor returns pointer-or-NULL, destructor
   is void, mutators return int rc. All go through `xgbcompat::check()`,
   which populates `g_last_error`.
2. Raw FFI in `xgboost-rkt/private/ffi-raw.rkt`: `define-cpointer-type`,
   deallocator-wrapped free, allocator-wrapped constructor, rc-returning
   mutators.
3. Wrapper in `xgboost-rkt/private/xgboost-native.rkt`: `contract-out`, NULL
   check after constructor, `check-ok` after mutators, tag-flip `*-free!`.
4. gtests in `cpp/tests/xgbcompat_test.cpp`: round-trip, error paths, leak
   smoke test with `getrusage`.
5. Racket tests in `(module+ test ...)`: round-trip, explicit-free+GC,
   double-free safety, shape-mismatch contract error, C-error surfacing, 10k
   create/free balance loop, finalizer-path loop.
