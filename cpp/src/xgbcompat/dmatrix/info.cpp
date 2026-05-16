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

int xgb_dmatrix_set_float_info(xgb_dmatrix_t handle, const char* field,
                               const float* values, size_t len) {
  if (!handle || !field || (!values && len > 0)) {
    xgbcompat::g_last_error = "xgb_dmatrix_set_float_info: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGDMatrixSetFloatInfo(static_cast<DMatrixHandle>(handle), field, values,
                              static_cast<bst_ulong>(len)),
        "XGDMatrixSetFloatInfo");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_set_float_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_set_uint_info(xgb_dmatrix_t handle, const char* field,
                              const uint32_t* values, size_t len) {
  if (!handle || !field || (!values && len > 0)) {
    xgbcompat::g_last_error = "xgb_dmatrix_set_uint_info: invalid argument";
    return 1;
  }
  try {
    static_assert(sizeof(uint32_t) == sizeof(unsigned),
                  "XGBoost uint info must use 32-bit unsigned values");
    xgbcompat::check(
        XGDMatrixSetUIntInfo(static_cast<DMatrixHandle>(handle), field,
                             reinterpret_cast<const unsigned*>(values),
                             static_cast<bst_ulong>(len)),
        "XGDMatrixSetUIntInfo");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_set_uint_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_set_info_from_interface(xgb_dmatrix_t handle, const char* field,
                                        const char* data) {
  if (!handle || !field || !data) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_info_from_interface: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(XGDMatrixSetInfoFromInterface(
                         static_cast<DMatrixHandle>(handle), field, data),
                     "XGDMatrixSetInfoFromInterface");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_info_from_interface: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_set_str_feature_info(xgb_dmatrix_t handle, const char* field,
                                     const char** values, size_t len) {
  if (!handle || !field || (!values && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_str_feature_info: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGDMatrixSetStrFeatureInfo(static_cast<DMatrixHandle>(handle), field,
                                   values, static_cast<bst_ulong>(len)),
        "XGDMatrixSetStrFeatureInfo");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_set_str_feature_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_str_feature_info(xgb_dmatrix_t handle, const char* field,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len, uint64_t* out_count) {
  if (!handle || !field || !out_len || !out_count ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_get_str_feature_info: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_count = 0;
  try {
    bst_ulong n = 0;
    const char** values = nullptr;
    xgbcompat::check(
        XGDMatrixGetStrFeatureInfo(static_cast<DMatrixHandle>(handle), field,
                                   &n, &values),
        "XGDMatrixGetStrFeatureInfo");
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
        "xgb_dmatrix_get_str_feature_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_float_info(xgb_dmatrix_t handle, const char* field,
                               uint64_t* out_len, const float** out_ptr) {
  if (!handle || !field || !out_len || !out_ptr) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_float_info: invalid argument";
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
    xgbcompat::g_last_error = "xgb_dmatrix_get_float_info: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_uint_info(xgb_dmatrix_t handle, const char* field,
                              uint64_t* out_len, const uint32_t** out_ptr) {
  if (!handle || !field || !out_len || !out_ptr) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_uint_info: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_ptr = nullptr;
  try {
    bst_ulong n = 0;
    const unsigned* data = nullptr;
    xgbcompat::check(XGDMatrixGetUIntInfo(static_cast<DMatrixHandle>(handle),
                                          field, &n, &data),
                     "XGDMatrixGetUIntInfo");
    *out_len = static_cast<uint64_t>(n);
    *out_ptr = reinterpret_cast<const uint32_t*>(data);
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_uint_info: unknown exception";
    return 1;
  }
}

}  // extern "C"
