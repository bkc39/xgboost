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

int xgb_booster_dump_model(xgb_booster_t handle, const char* format,
                           int with_stats, uint64_t buffer_capacity,
                           char* out_buffer, uint64_t* out_len,
                           uint64_t* out_count) {
  if (!handle || !format || !out_len || !out_count ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error = "xgb_booster_dump_model: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_count = 0;
  try {
    bst_ulong n = 0;
    const char** dumps = nullptr;
    xgbcompat::check(XGBoosterDumpModelEx(static_cast<BoosterHandle>(handle),
                                          "", with_stats, format, &n, &dumps),
                     "XGBoosterDumpModelEx");
    const uint64_t need = xgbcompat::nul_separated_size(dumps, n);
    *out_len = need;
    *out_count = static_cast<uint64_t>(n);
    if (buffer_capacity < need) {
      return 2;
    }
    if (need > 0 && out_buffer) {
      xgbcompat::copy_nul_separated(dumps, n, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_dump_model: unknown exception";
    return 1;
  }
}

int xgb_booster_dump_model_with_features(
    xgb_booster_t handle, const char** feature_names,
    const char** feature_types, size_t n_features, const char* format,
    int with_stats, uint64_t buffer_capacity, char* out_buffer,
    uint64_t* out_len, uint64_t* out_count) {
  if (!handle || !format || !out_len || !out_count ||
      (!feature_names && n_features > 0) ||
      (!feature_types && n_features > 0) ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_booster_dump_model_with_features: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_count = 0;
  try {
    bst_ulong n = 0;
    const char** dumps = nullptr;
    xgbcompat::check(
        XGBoosterDumpModelExWithFeatures(
            static_cast<BoosterHandle>(handle), static_cast<int>(n_features),
            feature_names, feature_types, with_stats, format, &n, &dumps),
        "XGBoosterDumpModelExWithFeatures");
    const uint64_t need = xgbcompat::nul_separated_size(dumps, n);
    *out_len = need;
    *out_count = static_cast<uint64_t>(n);
    if (buffer_capacity < need) {
      return 2;
    }
    if (need > 0 && out_buffer) {
      xgbcompat::copy_nul_separated(dumps, n, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_dump_model_with_features: unknown exception";
    return 1;
  }
}

}  // extern "C"
