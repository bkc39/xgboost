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

xgb_dmatrix_t xgb_dmatrix_create_from_mat(const float* data, size_t nrow,
                                          size_t ncol, float missing) {
  if (!data) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_mat: data pointer is null";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(
        XGDMatrixCreateFromMat(data, static_cast<bst_ulong>(nrow),
                               static_cast<bst_ulong>(ncol), missing, &handle),
        "XGDMatrixCreateFromMat");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_mat: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_uri(const char* config) {
  if (!config) {
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_uri: config is null";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(XGDMatrixCreateFromURI(config, &handle),
                     "XGDMatrixCreateFromURI");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_uri: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_dense(const char* data,
                                            const char* config) {
  if (!data || !config) {
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_dense: invalid argument";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(XGDMatrixCreateFromDense(data, config, &handle),
                     "XGDMatrixCreateFromDense");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_dense: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_csr(const char* indptr,
                                          const char* indices, const char* data,
                                          uint64_t ncol, const char* config) {
  if (!indptr || !indices || !data || !config) {
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_csr: invalid argument";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(
        XGDMatrixCreateFromCSR(indptr, indices, data,
                               static_cast<bst_ulong>(ncol), config, &handle),
        "XGDMatrixCreateFromCSR");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_csr: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_csc(const char* indptr,
                                          const char* indices, const char* data,
                                          uint64_t nrow, const char* config) {
  if (!indptr || !indices || !data || !config) {
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_csc: invalid argument";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(
        XGDMatrixCreateFromCSC(indptr, indices, data,
                               static_cast<bst_ulong>(nrow), config, &handle),
        "XGDMatrixCreateFromCSC");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error = "xgb_dmatrix_create_from_csc: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_create_from_columnar(const char* data,
                                               const char* config) {
  if (!data || !config) {
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_columnar: invalid argument";
    return nullptr;
  }
  DMatrixHandle handle = nullptr;
  try {
    xgbcompat::check(XGDMatrixCreateFromColumnar(data, config, &handle),
                     "XGDMatrixCreateFromColumnar");
    return static_cast<xgb_dmatrix_t>(handle);
  } catch (const std::exception&) {
    if (handle)
      XGDMatrixFree(handle);
    return nullptr;
  } catch (...) {
    if (handle)
      XGDMatrixFree(handle);
    xgbcompat::g_last_error =
        "xgb_dmatrix_create_from_columnar: unknown exception";
    return nullptr;
  }
}

xgb_dmatrix_t xgb_dmatrix_slice(xgb_dmatrix_t handle, const int32_t* indices,
                                size_t len, int allow_groups) {
  if (!handle || (!indices && len > 0)) {
    xgbcompat::g_last_error = "xgb_dmatrix_slice: invalid argument";
    return nullptr;
  }
  DMatrixHandle out = nullptr;
  try {
    static_assert(sizeof(int32_t) == sizeof(int),
                  "XGBoost row indices must be 32-bit ints");
    xgbcompat::check(XGDMatrixSliceDMatrixEx(
                         static_cast<DMatrixHandle>(handle),
                         reinterpret_cast<const int*>(indices),
                         static_cast<bst_ulong>(len), &out, allow_groups),
                     "XGDMatrixSliceDMatrixEx");
    return static_cast<xgb_dmatrix_t>(out);
  } catch (const std::exception&) {
    if (out)
      XGDMatrixFree(out);
    return nullptr;
  } catch (...) {
    if (out)
      XGDMatrixFree(out);
    xgbcompat::g_last_error = "xgb_dmatrix_slice: unknown exception";
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

}  // extern "C"
