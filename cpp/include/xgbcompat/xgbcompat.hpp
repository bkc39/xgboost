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

/* Opaque handle for an XGBoost Booster.  Same lifetime contract as the
 * DMatrix family: every successful create must be paired with exactly one
 * free; errors land in the thread-local `xgb_last_error()` buffer. */
typedef void* xgb_booster_t;

/* Create a booster, optionally seeding it with cache DMatrices used for
 * histogram construction.  `cache` may be NULL when `cache_len == 0`.
 * Returns NULL on failure. */
xgb_booster_t xgb_booster_create(const xgb_dmatrix_t* cache, size_t cache_len);

/* Release a booster.  Safe to call on NULL. */
void xgb_booster_free(xgb_booster_t handle);

/* Set a (string-valued) booster parameter, e.g. ("objective",
 * "reg:squarederror") or ("max_depth", "6"). */
int xgb_booster_set_param(xgb_booster_t handle,
                          const char* key,
                          const char* value);

/* Run one boosting round on `dtrain` at iteration `iter`. */
int xgb_booster_update_one_iter(xgb_booster_t handle,
                                int iter,
                                xgb_dmatrix_t dtrain);

/* Predict against a DMatrix and copy the result into a caller-owned buffer.
 *
 * `config` is an XGBoost JSON predict-config (pass "{}" for defaults — type 0,
 * inference mode, all iterations).  `buffer_capacity` is the number of floats
 * available in `out_buffer`; pass 0 / NULL to size-probe.  `*out_len` is
 * always written with the number of predictions XGBoost produced.
 *
 * Return values:
 *   0: success, prediction copied into out_buffer, out_len set
 *   1: error — call xgb_last_error()
 *   2: out_buffer too small; out_len holds the required size, nothing copied.
 *      The caller may grow the buffer and retry.
 *
 * Predictions are copied immediately, so the booster-owned buffer used by
 * XGBoost internally (which is invalidated by the next predict call) does
 * not escape. */
int xgb_booster_predict(xgb_booster_t handle,
                        xgb_dmatrix_t dmat,
                        const char* config,
                        uint64_t buffer_capacity,
                        float* out_buffer,
                        uint64_t* out_len);

/* Save the booster to a file.  XGBoost picks the format from the extension:
 * `.json` and `.ubj` are the supported portable formats. */
int xgb_booster_save_model(xgb_booster_t handle, const char* path);

/* Load a booster from a file written by `xgb_booster_save_model`. */
int xgb_booster_load_model(xgb_booster_t handle, const char* path);

/* Serialize the booster into a caller-owned byte buffer.
 *
 * `config` is a JSON string controlling the format, e.g. `{"format":"json"}`
 * or `{"format":"ubj"}` (the latter is the default UBJSON encoding).
 * Behaviour mirrors `xgb_booster_predict`:
 *   0: success, `*out_len` bytes written into `out_buffer`.
 *   1: error — call `xgb_last_error()`.
 *   2: buffer too small; `*out_len` holds the required size.  Resize and
 *      retry.  The booster-owned buffer is invalidated by the next save
 *      call, so we copy on every successful call. */
int xgb_booster_save_model_to_buffer(xgb_booster_t handle,
                                     const char* config,
                                     uint64_t buffer_capacity,
                                     char* out_buffer,
                                     uint64_t* out_len);

/* Load a booster from a byte buffer produced by save_model_to_buffer (or
 * an externally provided XGBoost JSON/UBJ blob). */
int xgb_booster_load_model_from_buffer(xgb_booster_t handle,
                                       const void* data,
                                       uint64_t len);

/* Evaluate the booster on one or more DMatrices, returning the metric line
 * XGBoost would otherwise print after each round.  `names` are user-chosen
 * labels for each entry of `dmats` (e.g. "train", "test"); they appear in
 * the result string.
 *
 * Output is a UTF-8 byte string like:
 *   "[0]\ttrain-rmse:1.234\ttest-rmse:1.456"
 *
 * Same size-then-fill contract as predict / save_model_to_buffer:
 *   0: success, `*out_len` bytes (excluding any null terminator) written
 *      into `out_buffer`.
 *   1: error — call `xgb_last_error()`.
 *   2: out_buffer too small; `*out_len` holds the required size.  Resize
 *      and retry.  Booster-owned buffer never escapes. */
int xgb_booster_eval_one_iter(xgb_booster_t handle,
                              int iter,
                              const xgb_dmatrix_t* dmats,
                              const char** names,
                              size_t n_dmats,
                              uint64_t buffer_capacity,
                              char* out_buffer,
                              uint64_t* out_len);

#ifdef __cplusplus
}
#endif
