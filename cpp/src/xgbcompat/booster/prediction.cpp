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

namespace {

int copy_prediction_result(const bst_ulong* out_shape, bst_ulong out_dim,
                           const float* out_result, uint64_t buffer_capacity,
                           float* out_buffer, uint64_t* out_len) {
  uint64_t total = 1;
  for (bst_ulong i = 0; i < out_dim; ++i) {
    total *= static_cast<uint64_t>(out_shape[i]);
  }
  return xgbcompat::copy_float_result(out_result, total, buffer_capacity,
                                      out_buffer, out_len);
}

}  // namespace

extern "C" {

int xgb_booster_predict(xgb_booster_t handle, xgb_dmatrix_t dmat,
                        const char* config, uint64_t buffer_capacity,
                        float* out_buffer, uint64_t* out_len) {
  if (!handle || !dmat || !config || !out_len) {
    xgbcompat::g_last_error = "xgb_booster_predict: invalid argument";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_booster_predict: capacity > 0 but out_buffer is null";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong const* out_shape = nullptr;
    bst_ulong out_dim = 0;
    float const* out_result = nullptr;
    xgbcompat::check(
        XGBoosterPredictFromDMatrix(static_cast<BoosterHandle>(handle),
                                    static_cast<DMatrixHandle>(dmat), config,
                                    &out_shape, &out_dim, &out_result),
        "XGBoosterPredictFromDMatrix");
    uint64_t total = 1;
    for (bst_ulong i = 0; i < out_dim; ++i) {
      total *= static_cast<uint64_t>(out_shape[i]);
    }
    *out_len = total;
    if (buffer_capacity < total) {
      // Booster-owned buffer is invalidated on the next predict call; we
      // must NOT hand out_result back to the caller.  Signal "too small" so
      // they can resize and retry.
      return 2;
    }
    if (total > 0 && out_result && out_buffer) {
      std::copy(out_result, out_result + total, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_predict: unknown exception";
    return 1;
  }
}

int xgb_booster_predict_from_dense(xgb_booster_t handle, const char* values,
                                   const char* config, xgb_dmatrix_t proxy,
                                   uint64_t buffer_capacity, float* out_buffer,
                                   uint64_t* out_len) {
  if (!handle || !values || !config || !out_len ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_booster_predict_from_dense: invalid argument";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong const* out_shape = nullptr;
    bst_ulong out_dim = 0;
    float const* out_result = nullptr;
    xgbcompat::check(
        XGBoosterPredictFromDense(static_cast<BoosterHandle>(handle), values,
                                  config, static_cast<DMatrixHandle>(proxy),
                                  &out_shape, &out_dim, &out_result),
        "XGBoosterPredictFromDense");
    return copy_prediction_result(out_shape, out_dim, out_result,
                                  buffer_capacity, out_buffer, out_len);
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_predict_from_dense: unknown exception";
    return 1;
  }
}

int xgb_booster_predict_from_csr(xgb_booster_t handle, const char* indptr,
                                 const char* indices, const char* values,
                                 uint64_t ncol, const char* config,
                                 xgb_dmatrix_t proxy, uint64_t buffer_capacity,
                                 float* out_buffer, uint64_t* out_len) {
  if (!handle || !indptr || !indices || !values || !config || !out_len ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error = "xgb_booster_predict_from_csr: invalid argument";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong const* out_shape = nullptr;
    bst_ulong out_dim = 0;
    float const* out_result = nullptr;
    xgbcompat::check(
        XGBoosterPredictFromCSR(static_cast<BoosterHandle>(handle), indptr,
                                indices, values, static_cast<bst_ulong>(ncol),
                                config, static_cast<DMatrixHandle>(proxy),
                                &out_shape, &out_dim, &out_result),
        "XGBoosterPredictFromCSR");
    return copy_prediction_result(out_shape, out_dim, out_result,
                                  buffer_capacity, out_buffer, out_len);
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_predict_from_csr: unknown exception";
    return 1;
  }
}

int xgb_booster_predict_from_columnar(xgb_booster_t handle, const char* values,
                                      const char* config, xgb_dmatrix_t proxy,
                                      uint64_t buffer_capacity,
                                      float* out_buffer, uint64_t* out_len) {
  if (!handle || !values || !config || !out_len ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_booster_predict_from_columnar: invalid argument";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong const* out_shape = nullptr;
    bst_ulong out_dim = 0;
    float const* out_result = nullptr;
    xgbcompat::check(
        XGBoosterPredictFromColumnar(static_cast<BoosterHandle>(handle), values,
                                     config, static_cast<DMatrixHandle>(proxy),
                                     &out_shape, &out_dim, &out_result),
        "XGBoosterPredictFromColumnar");
    return copy_prediction_result(out_shape, out_dim, out_result,
                                  buffer_capacity, out_buffer, out_len);
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_predict_from_columnar: unknown exception";
    return 1;
  }
}

}  // extern "C"
