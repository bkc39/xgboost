#include <xgboost/c_api.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "xgbcompat/detail/buffer.hpp"
#include "xgbcompat/detail/error.hpp"
#include "xgbcompat/detail/handles.hpp"
#include "xgbcompat/xgbcompat.hpp"

extern "C" {

int xgb_booster_eval_one_iter(xgb_booster_t handle, int iter,
                              const xgb_dmatrix_t* dmats, const char** names,
                              size_t n_dmats, uint64_t buffer_capacity,
                              char* out_buffer, uint64_t* out_len) {
  if (!handle || !out_len) {
    xgbcompat::g_last_error = "xgb_booster_eval_one_iter: invalid argument";
    return 1;
  }
  if (n_dmats > 0 && (!dmats || !names)) {
    xgbcompat::g_last_error =
        "xgb_booster_eval_one_iter: dmats/names null but n_dmats > 0";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_booster_eval_one_iter: capacity > 0 but out_buffer null";
    return 1;
  }
  *out_len = 0;
  try {
    static_assert(sizeof(DMatrixHandle) == sizeof(xgb_dmatrix_t),
                  "DMatrixHandle and xgb_dmatrix_t must be ABI-compatible");
    // XGBoosterEvalOneIter takes non-const dmats[]; const_cast is safe
    // because the call only reads handle values, never writes them.
    const char* result = nullptr;
    xgbcompat::check(
        XGBoosterEvalOneIter(
            static_cast<BoosterHandle>(handle), iter,
            reinterpret_cast<DMatrixHandle*>(const_cast<xgb_dmatrix_t*>(dmats)),
            names, static_cast<bst_ulong>(n_dmats), &result),
        "XGBoosterEvalOneIter");
    if (!result) {
      // Defensive: XGBoost should never return a null result on rc=0, but
      // if it ever does, treat as empty.
      return 0;
    }
    const size_t len = std::strlen(result);
    *out_len = static_cast<uint64_t>(len);
    if (buffer_capacity < *out_len) {
      // Booster-owned buffer is invalidated on the next eval call; never
      // hand it back across the FFI boundary.
      return 2;
    }
    if (len > 0 && out_buffer) {
      std::copy(result, result + len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_eval_one_iter: unknown exception";
    return 1;
  }
}

}  // extern "C"
