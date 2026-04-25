#include "xgbcompat/xgbcompat.hpp"
#include "internal.hpp"

#include <xgboost/c_api.h>

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

}  // extern "C"
