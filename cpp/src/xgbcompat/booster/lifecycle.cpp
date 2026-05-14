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

xgb_booster_t xgb_booster_create(const xgb_dmatrix_t* cache, size_t cache_len) {
  if (cache_len > 0 && !cache) {
    xgbcompat::g_last_error =
        "xgb_booster_create: cache pointer is null but cache_len > 0";
    return nullptr;
  }
  BoosterHandle handle = nullptr;
  try {
    // DMatrixHandle and xgb_dmatrix_t are both `void*`, so reinterpret is
    // safe; static_assert backstops any future drift.
    static_assert(sizeof(DMatrixHandle) == sizeof(xgb_dmatrix_t),
                  "DMatrixHandle and xgb_dmatrix_t must be ABI-compatible");
    xgbcompat::check(
        XGBoosterCreate(reinterpret_cast<const DMatrixHandle*>(cache),
                        static_cast<bst_ulong>(cache_len), &handle),
        "XGBoosterCreate");
    return static_cast<xgb_booster_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGBoosterFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGBoosterFree(handle);
    xgbcompat::g_last_error = "xgb_booster_create: unknown exception";
    return nullptr;
  }
}

void xgb_booster_free(xgb_booster_t handle) {
  if (handle) {
    XGBoosterFree(static_cast<BoosterHandle>(handle));
  }
}

int xgb_booster_reset(xgb_booster_t handle) {
  if (!handle) {
    xgbcompat::g_last_error = "xgb_booster_reset: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(XGBoosterReset(static_cast<BoosterHandle>(handle)),
                     "XGBoosterReset");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_reset: unknown exception";
    return 1;
  }
}

xgb_booster_t xgb_booster_slice(xgb_booster_t handle, int begin_layer,
                                int end_layer, int step) {
  if (!handle) {
    xgbcompat::g_last_error = "xgb_booster_slice: invalid argument";
    return nullptr;
  }
  BoosterHandle out = nullptr;
  try {
    xgbcompat::check(XGBoosterSlice(static_cast<BoosterHandle>(handle),
                                    begin_layer, end_layer, step, &out),
                     "XGBoosterSlice");
    return static_cast<xgb_booster_t>(out);
  } catch (const std::exception&) {
    if (out)
      XGBoosterFree(out);
    return nullptr;
  } catch (...) {
    if (out)
      XGBoosterFree(out);
    xgbcompat::g_last_error = "xgb_booster_slice: unknown exception";
    return nullptr;
  }
}

int xgb_booster_boosted_rounds(xgb_booster_t handle, int* out_rounds) {
  if (!handle || !out_rounds) {
    xgbcompat::g_last_error = "xgb_booster_boosted_rounds: invalid argument";
    return 1;
  }
  try {
    int rounds = 0;
    xgbcompat::check(
        XGBoosterBoostedRounds(static_cast<BoosterHandle>(handle), &rounds),
        "XGBoosterBoostedRounds");
    *out_rounds = rounds;
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_boosted_rounds: unknown exception";
    return 1;
  }
}

int xgb_booster_num_feature(xgb_booster_t handle, uint64_t* out_features) {
  if (!handle || !out_features) {
    xgbcompat::g_last_error = "xgb_booster_num_feature: invalid argument";
    return 1;
  }
  try {
    bst_ulong n = 0;
    xgbcompat::check(
        XGBoosterGetNumFeature(static_cast<BoosterHandle>(handle), &n),
        "XGBoosterGetNumFeature");
    *out_features = static_cast<uint64_t>(n);
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_num_feature: unknown exception";
    return 1;
  }
}

}  // extern "C"
