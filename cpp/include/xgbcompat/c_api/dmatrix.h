#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for an XGBoost DMatrix.  The caller is responsible for
 * pairing each successful `xgb_dmatrix_create_*` with exactly one
 * `xgb_dmatrix_free`.  Errors set the thread-local buffer read by
 * `xgb_last_error()`. */
typedef void* xgb_dmatrix_t;

/* Create a DMatrix from a row-major float array of length `nrow * ncol`.
 * Returns NULL on failure; call `xgb_last_error()` for the message.
 * `missing` is the sentinel treated as a missing value by XGBoost. */
xgb_dmatrix_t xgb_dmatrix_create_from_mat(const float* data, size_t nrow,
                                          size_t ncol, float missing);

/* Create a DMatrix from a URI config JSON string.  The config is passed
 * directly to XGDMatrixCreateFromURI, e.g.
 *   {"uri":"/path/to/data.libsvm?format=libsvm","silent":true} */
xgb_dmatrix_t xgb_dmatrix_create_from_uri(const char* config);

/* Create DMatrices from XGBoost's JSON encoded array-interface inputs.
 * These wrappers return NULL on failure; call xgb_last_error() for details.
 * The C API copies inputs during construction, so the JSON strings and
 * referenced buffers only need to stay valid for the duration of the call. */
xgb_dmatrix_t xgb_dmatrix_create_from_dense(const char* data,
                                            const char* config);
xgb_dmatrix_t xgb_dmatrix_create_from_csr(const char* indptr,
                                          const char* indices, const char* data,
                                          uint64_t ncol, const char* config);
xgb_dmatrix_t xgb_dmatrix_create_from_csc(const char* indptr,
                                          const char* indices, const char* data,
                                          uint64_t nrow, const char* config);
xgb_dmatrix_t xgb_dmatrix_create_from_columnar(const char* data,
                                               const char* config);

/* Create a new DMatrix by selecting row indices from an existing DMatrix. */
xgb_dmatrix_t xgb_dmatrix_slice(xgb_dmatrix_t handle, const int32_t* indices,
                                size_t len, int allow_groups);

/* Release a DMatrix.  Safe to call on NULL. */
void xgb_dmatrix_free(xgb_dmatrix_t handle);

/* Attach a float-valued info field (e.g. "label", "weight") to a DMatrix.
 * Returns 0 on success, non-zero on failure (check `xgb_last_error()`). */
int xgb_dmatrix_set_float_info(xgb_dmatrix_t handle, const char* field,
                               const float* values, size_t len);

/* Attach unsigned integer metadata, e.g. ranking group/qid fields. */
int xgb_dmatrix_set_uint_info(xgb_dmatrix_t handle, const char* field,
                              const uint32_t* values, size_t len);

/* Attach metadata using XGBoost's JSON array-interface path. */
int xgb_dmatrix_set_info_from_interface(xgb_dmatrix_t handle, const char* field,
                                        const char* data);

/* Attach feature names or types.  Accepted fields are "feature_name" and
 * "feature_type". */
int xgb_dmatrix_set_str_feature_info(xgb_dmatrix_t handle, const char* field,
                                     const char** values, size_t len);

/* Copy feature names or types into a NUL-separated UTF-8 buffer.
 *
 * Return values mirror other size-then-fill APIs:
 *   0: success
 *   1: error — call xgb_last_error()
 *   2: out_buffer too small; out_len has the required byte count
 */
int xgb_dmatrix_get_str_feature_info(xgb_dmatrix_t handle, const char* field,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len, uint64_t* out_count);

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
int xgb_dmatrix_get_float_info(xgb_dmatrix_t handle, const char* field,
                               uint64_t* out_len, const float** out_ptr);

/* Read back a uint32-valued info field.  The pointer is borrowed from
 * XGBoost; callers should copy before making another XGBoost API call. */
int xgb_dmatrix_get_uint_info(xgb_dmatrix_t handle, const char* field,
                              uint64_t* out_len, const uint32_t** out_ptr);

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
int xgb_dmatrix_get_data_as_csr(xgb_dmatrix_t handle, const char* config,
                                uint64_t* out_indptr, uint32_t* out_indices,
                                float* out_data);

/* Save the DMatrix to a binary file readable via
 * `xgb_dmatrix_create_from_uri` with `format=binary`.
 * `silent` is forwarded to XGBoost; pass 1 to suppress its stdout chatter. */
int xgb_dmatrix_save_binary(xgb_dmatrix_t handle, const char* fname,
                            int silent);

/* Export the quantile cuts used by histogram-based training.  XGBoost
 * returns the indptr and data arrays as JSON-encoded `__array_interface__`
 * strings; callers parse them.  Both buffers are filled in a single call so
 * the thread-local outputs are read exactly once.
 *
 * Behaviour:
 *   0: both buffers were sized correctly.  `*out_indptr_len` and
 *      `*out_data_len` hold the bytes written.
 *   1: error — call `xgb_last_error()`.
 *   2: at least one buffer too small.  `*out_indptr_len` /
 *      `*out_data_len` hold the required sizes; nothing copied. */
int xgb_dmatrix_get_quantile_cut(xgb_dmatrix_t handle, const char* config,
                                 uint64_t indptr_capacity, char* out_indptr,
                                 uint64_t* out_indptr_len,
                                 uint64_t data_capacity, char* out_data,
                                 uint64_t* out_data_len);

#ifdef __cplusplus
}
#endif
