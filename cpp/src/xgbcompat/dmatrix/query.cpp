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

int xgb_dmatrix_num_non_missing(xgb_dmatrix_t handle, uint64_t* out_nnz) {
  if (!handle || !out_nnz) {
    xgbcompat::g_last_error = "xgb_dmatrix_num_non_missing: invalid argument";
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
    xgbcompat::g_last_error = "xgb_dmatrix_num_non_missing: unknown exception";
    return 1;
  }
}

int xgb_dmatrix_get_data_as_csr(xgb_dmatrix_t handle, const char* config,
                                uint64_t* out_indptr, uint32_t* out_indices,
                                float* out_data) {
  if (!handle || !config || !out_indptr || !out_indices || !out_data) {
    xgbcompat::g_last_error = "xgb_dmatrix_get_data_as_csr: invalid argument";
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
    xgbcompat::g_last_error = "xgb_dmatrix_get_data_as_csr: unknown exception";
    return 1;
  }
}

}  // extern "C"
