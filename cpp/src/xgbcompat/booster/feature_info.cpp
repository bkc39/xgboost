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

int xgb_booster_set_str_feature_info(xgb_booster_t handle, const char* field,
                                     const char** values, size_t len) {
  if (!handle || !field || (!values && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_booster_set_str_feature_info: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSetStrFeatureInfo(static_cast<BoosterHandle>(handle), field,
                                   values, static_cast<bst_ulong>(len)),
        "XGBoosterSetStrFeatureInfo");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_set_str_feature_info: unknown exception";
    return 1;
  }
}

int xgb_booster_get_str_feature_info(xgb_booster_t handle, const char* field,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len, uint64_t* out_count) {
  if (!handle || !field || !out_len || !out_count ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_booster_get_str_feature_info: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_count = 0;
  try {
    bst_ulong n = 0;
    const char** values = nullptr;
    xgbcompat::check(
        XGBoosterGetStrFeatureInfo(static_cast<BoosterHandle>(handle), field,
                                   &n, &values),
        "XGBoosterGetStrFeatureInfo");
    const uint64_t need = xgbcompat::nul_separated_size(values, n);
    *out_len = need;
    *out_count = static_cast<uint64_t>(n);
    if (buffer_capacity < need) {
      return 2;
    }
    if (need > 0 && out_buffer) {
      xgbcompat::copy_nul_separated(values, n, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_get_str_feature_info: unknown exception";
    return 1;
  }
}

int xgb_booster_feature_score(xgb_booster_t handle, const char* config,
                              uint64_t feature_capacity, char* out_features,
                              uint64_t* out_feature_len,
                              uint64_t* out_n_features, uint64_t shape_capacity,
                              uint64_t* out_shape, uint64_t* out_dim,
                              uint64_t score_capacity, float* out_scores,
                              uint64_t* out_n_scores) {
  if (!handle || !config || !out_feature_len || !out_n_features || !out_dim ||
      !out_n_scores || (feature_capacity > 0 && !out_features) ||
      (shape_capacity > 0 && !out_shape) ||
      (score_capacity > 0 && !out_scores)) {
    xgbcompat::g_last_error = "xgb_booster_feature_score: invalid argument";
    return 1;
  }
  *out_feature_len = 0;
  *out_n_features = 0;
  *out_dim = 0;
  *out_n_scores = 0;
  try {
    bst_ulong n_features = 0;
    const char** features = nullptr;
    bst_ulong dim = 0;
    const bst_ulong* shape = nullptr;
    const float* scores = nullptr;
    xgbcompat::check(
        XGBoosterFeatureScore(static_cast<BoosterHandle>(handle), config,
                              &n_features, &features, &dim, &shape, &scores),
        "XGBoosterFeatureScore");

    const uint64_t feature_bytes =
        xgbcompat::nul_separated_size(features, n_features);
    uint64_t n_scores = 1;
    for (bst_ulong i = 0; i < dim; ++i) {
      n_scores *= static_cast<uint64_t>(shape[i]);
    }
    if (dim == 0) {
      n_scores = 0;
    }

    *out_feature_len = feature_bytes;
    *out_n_features = static_cast<uint64_t>(n_features);
    *out_dim = static_cast<uint64_t>(dim);
    *out_n_scores = n_scores;

    if (feature_capacity < feature_bytes ||
        shape_capacity < static_cast<uint64_t>(dim) ||
        score_capacity < n_scores) {
      return 2;
    }
    if (feature_bytes > 0 && out_features) {
      xgbcompat::copy_nul_separated(features, n_features, out_features);
    }
    if (dim > 0 && out_shape) {
      for (bst_ulong i = 0; i < dim; ++i) {
        out_shape[i] = static_cast<uint64_t>(shape[i]);
      }
    }
    if (n_scores > 0 && scores && out_scores) {
      std::copy(scores, scores + n_scores, out_scores);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_feature_score: unknown exception";
    return 1;
  }
}

}  // extern "C"
