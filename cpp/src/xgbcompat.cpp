#include "xgbcompat/xgbcompat.hpp"
#include "internal.hpp"

#include <xgboost/c_api.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>

namespace xgbcompat {
namespace {

thread_local std::string g_version_cache;

}  // namespace
}  // namespace xgbcompat

extern "C" {

const char* xgb_version(void) {
  try {
    xgbcompat::g_version_cache = xgbcompat::version();
    return xgbcompat::g_version_cache.c_str();
  } catch (...) {
    return "";
  }
}

const char* xgb_last_error(void) {
  static thread_local std::string buf;
  buf = xgbcompat::last_error();
  return buf.c_str();
}

int xgb_run_regression_demo(double* out_first_prediction) {
  try {
    const auto result = xgbcompat::run_regression_demo();
    if (result.predictions.empty()) return 2;
    if (out_first_prediction) {
      *out_first_prediction = static_cast<double>(result.predictions.front());
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    return 1;
  }
}

int xgb_run_classification_demo(double* out_first_prediction) {
  try {
    const auto result = xgbcompat::run_classification_demo();
    if (result.predictions.empty()) return 2;
    if (out_first_prediction) {
      *out_first_prediction = static_cast<double>(result.predictions.front());
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    return 1;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_mat(const float* data,
                                          size_t nrow,
                                          size_t ncol,
                                          float missing) {
  if (!data) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_mat: data pointer is null";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(XGDMatrixCreateFromMat(data,
                                            static_cast<bst_ulong>(nrow),
                                            static_cast<bst_ulong>(ncol),
                                            missing, &handle),
                     "XGDMatrixCreateFromMat");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle) XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle) XGDMatrixFree(handle);
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_mat: unknown exception";
    return nullptr;
  }
}

void xgb_dmatrix_free(xgb_dmatrix_t handle) {
  if (handle) {
    // XGDMatrixFree returns an rc, but we have nowhere to surface it on a
    // destructor path; swallow and never throw.
    XGDMatrixFree(static_cast<DMatrixHandle>(handle));
  }
}

int xgb_dmatrix_set_float_info(xgb_dmatrix_t handle,
                               const char* field,
                               const float* values,
                               size_t len) {
  if (!handle || !field || (!values && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_float_info: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(XGDMatrixSetFloatInfo(static_cast<DMatrixHandle>(handle),
                                           field, values,
                                           static_cast<bst_ulong>(len)),
                     "XGDMatrixSetFloatInfo");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_float_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_num_row(xgb_dmatrix_t handle, uint64_t* out_nrow) {
  if (!handle || !out_nrow) {
    xgbcompat::g_last_error = "xgb_dmatrix_num_row: invalid argument";
    return 1;
  }
  try {
    bst_ulong n = 0;
    xgbcompat::check(XGDMatrixNumRow(static_cast<DMatrixHandle>(handle), &n),
                     "XGDMatrixNumRow");
    *out_nrow = static_cast<uint64_t>(n);
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_num_row: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_num_col(xgb_dmatrix_t handle, uint64_t* out_ncol) {
  if (!handle || !out_ncol) {
    xgbcompat::g_last_error = "xgb_dmatrix_num_col: invalid argument";
    return 1;
  }
  try {
    bst_ulong n = 0;
    xgbcompat::check(XGDMatrixNumCol(static_cast<DMatrixHandle>(handle), &n),
                     "XGDMatrixNumCol");
    *out_ncol = static_cast<uint64_t>(n);
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_num_col: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_float_info(xgb_dmatrix_t handle,
                               const char* field,
                               uint64_t* out_len,
                               const float** out_ptr) {
  if (!handle || !field || !out_len || !out_ptr) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_get_float_info: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_ptr = nullptr;
  try {
    bst_ulong n = 0;
    const float* data = nullptr;
    xgbcompat::check(XGDMatrixGetFloatInfo(static_cast<DMatrixHandle>(handle),
                                           field, &n, &data),
                     "XGDMatrixGetFloatInfo");
    *out_len = static_cast<uint64_t>(n);
    *out_ptr = data;
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_get_float_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_num_non_missing(xgb_dmatrix_t handle, uint64_t* out_nnz) {
  if (!handle || !out_nnz) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_num_non_missing: invalid argument";
    return 1;
  }
  try {
    bst_ulong n = 0;
    xgbcompat::check(
        XGDMatrixNumNonMissing(static_cast<DMatrixHandle>(handle), &n),
        "XGDMatrixNumNonMissing");
    *out_nnz = static_cast<uint64_t>(n);
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_num_non_missing: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_data_as_csr(xgb_dmatrix_t handle,
                                const char* config,
                                uint64_t* out_indptr,
                                uint32_t* out_indices,
                                float* out_data) {
  if (!handle || !config || !out_indptr || !out_indices || !out_data) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_get_data_as_csr: invalid argument";
    return 1;
  }
  try {
    // bst_ulong is uint64_t on all current platforms, so reinterpret_cast is
    // safe; static_assert catches any future divergence.
    static_assert(sizeof(bst_ulong) == sizeof(uint64_t),
                  "bst_ulong must be 64-bit for this cast");
    xgbcompat::check(
        XGDMatrixGetDataAsCSR(static_cast<DMatrixHandle>(handle), config,
                              reinterpret_cast<bst_ulong*>(out_indptr),
                              out_indices, out_data),
        "XGDMatrixGetDataAsCSR");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_get_data_as_csr: unknown exception";
    return 1;
  }
}

xgb_booster_t xgb_booster_create(const xgb_dmatrix_t* cache,
                                 size_t cache_len) {
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
    if (handle) XGBoosterFree(handle);
    return nullptr;
  } catch (...) {
    if (handle) XGBoosterFree(handle);
    xgbcompat::g_last_error = "xgb_booster_create: unknown exception";
    return nullptr;
  }
}

void xgb_booster_free(xgb_booster_t handle) {
  if (handle) {
    XGBoosterFree(static_cast<BoosterHandle>(handle));
  }
}

int xgb_booster_set_param(xgb_booster_t handle,
                          const char* key,
                          const char* value) {
  if (!handle || !key || !value) {
    xgbcompat::g_last_error = "xgb_booster_set_param: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(XGBoosterSetParam(static_cast<BoosterHandle>(handle),
                                       key, value),
                     "XGBoosterSetParam");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_set_param: unknown exception";
    return 1;
  }
}

int xgb_booster_update_one_iter(xgb_booster_t handle,
                                int iter,
                                xgb_dmatrix_t dtrain) {
  if (!handle || !dtrain) {
    xgbcompat::g_last_error = "xgb_booster_update_one_iter: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterUpdateOneIter(static_cast<BoosterHandle>(handle), iter,
                               static_cast<DMatrixHandle>(dtrain)),
        "XGBoosterUpdateOneIter");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_update_one_iter: unknown exception";
    return 1;
  }
}

int xgb_booster_predict(xgb_booster_t handle,
                        xgb_dmatrix_t dmat,
                        const char* config,
                        uint64_t buffer_capacity,
                        float* out_buffer,
                        uint64_t* out_len) {
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

int xgb_booster_save_model(xgb_booster_t handle, const char* path) {
  if (!handle || !path) {
    xgbcompat::g_last_error = "xgb_booster_save_model: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSaveModel(static_cast<BoosterHandle>(handle), path),
        "XGBoosterSaveModel");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_save_model: unknown exception";
    return 1;
  }
}

int xgb_booster_load_model(xgb_booster_t handle, const char* path) {
  if (!handle || !path) {
    xgbcompat::g_last_error = "xgb_booster_load_model: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterLoadModel(static_cast<BoosterHandle>(handle), path),
        "XGBoosterLoadModel");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_load_model: unknown exception";
    return 1;
  }
}

int xgb_booster_save_model_to_buffer(xgb_booster_t handle,
                                     const char* config,
                                     uint64_t buffer_capacity,
                                     char* out_buffer,
                                     uint64_t* out_len) {
  if (!handle || !config || !out_len) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: invalid argument";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: capacity > 0 but out_buffer null";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong n = 0;
    const char* data = nullptr;
    xgbcompat::check(
        XGBoosterSaveModelToBuffer(static_cast<BoosterHandle>(handle),
                                   config, &n, &data),
        "XGBoosterSaveModelToBuffer");
    *out_len = static_cast<uint64_t>(n);
    if (buffer_capacity < *out_len) {
      // Booster-owned buffer is invalidated on the next call; do not leak.
      return 2;
    }
    if (*out_len > 0 && data && out_buffer) {
      std::copy(data, data + *out_len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_load_model_from_buffer(xgb_booster_t handle,
                                       const void* data,
                                       uint64_t len) {
  if (!handle || (!data && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_booster_load_model_from_buffer: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterLoadModelFromBuffer(static_cast<BoosterHandle>(handle),
                                     data, static_cast<bst_ulong>(len)),
        "XGBoosterLoadModelFromBuffer");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_load_model_from_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_eval_one_iter(xgb_booster_t handle,
                              int iter,
                              const xgb_dmatrix_t* dmats,
                              const char** names,
                              size_t n_dmats,
                              uint64_t buffer_capacity,
                              char* out_buffer,
                              uint64_t* out_len) {
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
        XGBoosterEvalOneIter(static_cast<BoosterHandle>(handle), iter,
                             reinterpret_cast<DMatrixHandle*>(
                                 const_cast<xgb_dmatrix_t*>(dmats)),
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
    xgbcompat::g_last_error =
        "xgb_booster_eval_one_iter: unknown exception";
    return 1;
  }
}

}  // extern "C"
