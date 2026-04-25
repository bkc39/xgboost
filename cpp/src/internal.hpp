#pragma once

#include <string>

namespace xgbcompat {

// Thread-local buffer for the most recent FFI error message.  Populated by
// `check()` whenever an XGBoost C API call returns a non-zero rc, and surfaced
// to Racket/C callers via `xgb_last_error()`.
extern thread_local std::string g_last_error;

// Wrap an XGBoost C API call: if `rc != 0`, records a contextual message in
// `g_last_error` and throws `std::runtime_error`.  `where` should name the API
// call (e.g. "XGDMatrixCreateFromMat") so error messages are self-describing.
void check(int rc, const char* where);

}  // namespace xgbcompat
