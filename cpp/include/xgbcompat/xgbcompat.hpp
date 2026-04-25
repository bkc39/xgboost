#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace xgbcompat {

struct DemoResult {
  std::vector<float> predictions;
};

std::string version();

DemoResult run_regression_demo();

DemoResult run_classification_demo();

std::string last_error();

}  // namespace xgbcompat

#ifdef __cplusplus
extern "C" {
#endif

const char* xgb_version(void);

const char* xgb_last_error(void);

int xgb_run_regression_demo(double* out_first_prediction);

int xgb_run_classification_demo(double* out_first_prediction);

/* Opaque handle for an XGBoost DMatrix.  The caller is responsible for
 * pairing each successful `xgb_dmatrix_create_*` with exactly one
 * `xgb_dmatrix_free`.  Errors set the thread-local buffer read by
 * `xgb_last_error()`. */
typedef void* xgb_dmatrix_t;

/* Create a DMatrix from a row-major float array of length `nrow * ncol`.
 * Returns NULL on failure; call `xgb_last_error()` for the message.
 * `missing` is the sentinel treated as a missing value by XGBoost. */
xgb_dmatrix_t xgb_dmatrix_create_from_mat(const float* data,
                                          size_t nrow,
                                          size_t ncol,
                                          float missing);

/* Release a DMatrix.  Safe to call on NULL. */
void xgb_dmatrix_free(xgb_dmatrix_t handle);

/* Attach a float-valued info field (e.g. "label", "weight") to a DMatrix.
 * Returns 0 on success, non-zero on failure (check `xgb_last_error()`). */
int xgb_dmatrix_set_float_info(xgb_dmatrix_t handle,
                               const char* field,
                               const float* values,
                               size_t len);

/* Query the number of rows in a DMatrix.
 * Returns 0 on success and writes the count to `*out_nrow`. */
int xgb_dmatrix_num_row(xgb_dmatrix_t handle, uint64_t* out_nrow);

/* Query the number of columns in a DMatrix. */
int xgb_dmatrix_num_col(xgb_dmatrix_t handle, uint64_t* out_ncol);

/* Read back a float-valued info field.  On success writes the length to
 * `*out_len` and a pointer to the DMatrix-owned data buffer to `*out_ptr`.
 * The pointer is borrowed: valid only until the next mutation of the
 * DMatrix's info, and never outlives the DMatrix itself.  Callers should
 * copy the data before doing anything else with the handle. */
int xgb_dmatrix_get_float_info(xgb_dmatrix_t handle,
                               const char* field,
                               uint64_t* out_len,
                               const float** out_ptr);

/* Count non-missing entries in the DMatrix.  Needed to size the buffers
 * passed to `xgb_dmatrix_get_data_as_csr`. */
int xgb_dmatrix_num_non_missing(xgb_dmatrix_t handle, uint64_t* out_nnz);

/* Read the DMatrix back as a CSR triple.  `config` is an XGBoost JSON
 * config string; pass "{}" for defaults.  The three output buffers are
 * caller-owned and must be sized:
 *   out_indptr:  nrow + 1 elements
 *   out_indices: nnz elements (from xgb_dmatrix_num_non_missing)
 *   out_data:    nnz elements
 * For a DMatrix created with no missing values, nnz == nrow * ncol and
 * the data buffer contains the row-major feature matrix. */
int xgb_dmatrix_get_data_as_csr(xgb_dmatrix_t handle,
                                const char* config,
                                uint64_t* out_indptr,
                                uint32_t* out_indices,
                                float* out_data);

#ifdef __cplusplus
}
#endif
