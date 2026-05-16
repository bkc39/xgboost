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

int xgb_dmatrix_save_binary(xgb_dmatrix_t handle, const char* path,
                            int silent) {
  if (!handle || !path) {
    xgbcompat::g_last_error = "xgb_dmatrix_save_binary: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGDMatrixSaveBinary(static_cast<DMatrixHandle>(handle), path, silent),
        "XGDMatrixSaveBinary");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_save_binary: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_quantile_cut(xgb_dmatrix_t handle, const char* config,
                                 uint64_t indptr_capacity, char* out_indptr,
                                 uint64_t* out_indptr_len,
                                 uint64_t data_capacity, char* out_data,
                                 uint64_t* out_data_len) {
  if (!handle || !config || !out_indptr_len || !out_data_len ||
      (indptr_capacity > 0 && !out_indptr) ||
      (data_capacity > 0 && !out_data)) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_quantile_cut: invalid argument";
    return 1;
  }
  *out_indptr_len = 0;
  *out_data_len = 0;
  try {
    const char* indptr = nullptr;
    const char* data = nullptr;
    xgbcompat::check(XGDMatrixGetQuantileCut(static_cast<DMatrixHandle>(handle),
                                             config, &indptr, &data),
                     "XGDMatrixGetQuantileCut");
    const uint64_t indptr_len =
        static_cast<uint64_t>(indptr ? std::strlen(indptr) : 0);
    const uint64_t data_len =
        static_cast<uint64_t>(data ? std::strlen(data) : 0);
    *out_indptr_len = indptr_len;
    *out_data_len = data_len;
    if (indptr_capacity < indptr_len || data_capacity < data_len) {
      return 2;
    }
    if (indptr_len > 0 && out_indptr) {
      std::copy(indptr, indptr + indptr_len, out_indptr);
    }
    if (data_len > 0 && out_data) {
      std::copy(data, data + data_len, out_data);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_quantile_cut: unknown exception";
    return 1;
  }
}

}  // extern "C"
