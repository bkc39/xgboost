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

/* Return XGBoost build information as a UTF-8 JSON string.
 *
 * Same size-then-fill contract as booster prediction/string APIs:
 *   0: success, `*out_len` bytes written into `out_buffer`
 *   1: error — call `xgb_last_error()`
 *   2: out_buffer too small; `*out_len` holds the required size */
int xgb_build_info(uint64_t buffer_capacity,
                   char* out_buffer,
                   uint64_t* out_len);

/* Read and write XGBoost's process-global JSON configuration. */
int xgb_get_global_config(uint64_t buffer_capacity,
                          char* out_buffer,
                          uint64_t* out_len);
int xgb_set_global_config(const char* config);

/* Register a process-global XGBoost log callback.  The caller owns the
 * function pointer lifetime and must keep it alive for as long as XGBoost may
 * invoke it. */
int xgb_register_log_callback(void (*callback)(const char*));

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
                                          const char* indices,
                                          const char* data,
                                          uint64_t ncol,
                                          const char* config);
xgb_dmatrix_t xgb_dmatrix_create_from_csc(const char* indptr,
                                          const char* indices,
                                          const char* data,
                                          uint64_t nrow,
                                          const char* config);
xgb_dmatrix_t xgb_dmatrix_create_from_columnar(const char* data,
                                               const char* config);

/* Create a new DMatrix by selecting row indices from an existing DMatrix. */
xgb_dmatrix_t xgb_dmatrix_slice(xgb_dmatrix_t handle,
                                const int32_t* indices,
                                size_t len,
                                int allow_groups);

/* Release a DMatrix.  Safe to call on NULL. */
void xgb_dmatrix_free(xgb_dmatrix_t handle);

/* Attach a float-valued info field (e.g. "label", "weight") to a DMatrix.
 * Returns 0 on success, non-zero on failure (check `xgb_last_error()`). */
int xgb_dmatrix_set_float_info(xgb_dmatrix_t handle,
                               const char* field,
                               const float* values,
                               size_t len);

/* Attach unsigned integer metadata, e.g. ranking group/qid fields. */
int xgb_dmatrix_set_uint_info(xgb_dmatrix_t handle,
                              const char* field,
                              const uint32_t* values,
                              size_t len);

/* Attach metadata using XGBoost's JSON array-interface path. */
int xgb_dmatrix_set_info_from_interface(xgb_dmatrix_t handle,
                                        const char* field,
                                        const char* data);

/* Attach feature names or types.  Accepted fields are "feature_name" and
 * "feature_type". */
int xgb_dmatrix_set_str_feature_info(xgb_dmatrix_t handle,
                                     const char* field,
                                     const char** values,
                                     size_t len);

/* Copy feature names or types into a NUL-separated UTF-8 buffer.
 *
 * Return values mirror other size-then-fill APIs:
 *   0: success
 *   1: error — call xgb_last_error()
 *   2: out_buffer too small; out_len has the required byte count
 */
int xgb_dmatrix_get_str_feature_info(xgb_dmatrix_t handle,
                                     const char* field,
                                     uint64_t buffer_capacity,
                                     char* out_buffer,
                                     uint64_t* out_len,
                                     uint64_t* out_count);

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

/* Read back a uint32-valued info field.  The pointer is borrowed from
 * XGBoost; callers should copy before making another XGBoost API call. */
int xgb_dmatrix_get_uint_info(xgb_dmatrix_t handle,
                              const char* field,
                              uint64_t* out_len,
                              const uint32_t** out_ptr);

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

/* Save the DMatrix to a binary file readable via
 * `xgb_dmatrix_create_from_uri` with `format=binary`.
 * `silent` is forwarded to XGBoost; pass 1 to suppress its stdout chatter. */
int xgb_dmatrix_save_binary(xgb_dmatrix_t handle,
                            const char* fname,
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
int xgb_dmatrix_get_quantile_cut(xgb_dmatrix_t handle,
                                 const char* config,
                                 uint64_t indptr_capacity,
                                 char* out_indptr,
                                 uint64_t* out_indptr_len,
                                 uint64_t data_capacity,
                                 char* out_data,
                                 uint64_t* out_data_len);

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

/* Release training caches held by a booster. */
int xgb_booster_reset(xgb_booster_t handle);

/* Slice boosted rounds from a booster into a new booster. */
xgb_booster_t xgb_booster_slice(xgb_booster_t handle,
                                int begin_layer,
                                int end_layer,
                                int step);

/* Query booster structure. */
int xgb_booster_boosted_rounds(xgb_booster_t handle, int* out_rounds);
int xgb_booster_num_feature(xgb_booster_t handle, uint64_t* out_features);

/* Set a (string-valued) booster parameter, e.g. ("objective",
 * "reg:squarederror") or ("max_depth", "6"). */
int xgb_booster_set_param(xgb_booster_t handle,
                          const char* key,
                          const char* value);

/* Run one boosting round on `dtrain` at iteration `iter`. */
int xgb_booster_update_one_iter(xgb_booster_t handle,
                                int iter,
                                xgb_dmatrix_t dtrain);

/* Run one custom-objective boosting round using caller-provided gradient and
 * Hessian array-interface JSON strings. */
int xgb_booster_train_one_iter(xgb_booster_t handle,
                               xgb_dmatrix_t dtrain,
                               int iter,
                               const char* grad,
                               const char* hess);

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

/* CPU inplace prediction from array-interface inputs.  These mirror
 * `xgb_booster_predict`: XGBoost-owned prediction memory is copied into
 * caller-owned buffers, and rc=2 reports the required float count. */
int xgb_booster_predict_from_dense(xgb_booster_t handle,
                                   const char* values,
                                   const char* config,
                                   xgb_dmatrix_t proxy,
                                   uint64_t buffer_capacity,
                                   float* out_buffer,
                                   uint64_t* out_len);
int xgb_booster_predict_from_csr(xgb_booster_t handle,
                                 const char* indptr,
                                 const char* indices,
                                 const char* values,
                                 uint64_t ncol,
                                 const char* config,
                                 xgb_dmatrix_t proxy,
                                 uint64_t buffer_capacity,
                                 float* out_buffer,
                                 uint64_t* out_len);
int xgb_booster_predict_from_columnar(xgb_booster_t handle,
                                      const char* values,
                                      const char* config,
                                      xgb_dmatrix_t proxy,
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

/* Memory snapshot serialization.  Unlike save_model_to_buffer, this captures
 * the full booster state (training caches, iteration counters, etc.) so a
 * snapshot taken mid-training can resume `update_one_iter` calls in a fresh
 * booster.  Same size-then-fill contract as save_model_to_buffer. */
int xgb_booster_serialize_to_buffer(xgb_booster_t handle,
                                    uint64_t buffer_capacity,
                                    char* out_buffer,
                                    uint64_t* out_len);

/* Restore a booster from a snapshot produced by serialize_to_buffer. */
int xgb_booster_unserialize_from_buffer(xgb_booster_t handle,
                                        const void* data,
                                        uint64_t len);

/* Save/load XGBoost's internal JSON config. */
int xgb_booster_save_json_config(xgb_booster_t handle,
                                 uint64_t buffer_capacity,
                                 char* out_buffer,
                                 uint64_t* out_len);
int xgb_booster_load_json_config(xgb_booster_t handle, const char* config);

/* Booster attributes.  `get` uses a size-then-fill contract and reports
 * `found=0` for missing attrs. */
int xgb_booster_set_attr(xgb_booster_t handle,
                         const char* key,
                         const char* value);
int xgb_booster_delete_attr(xgb_booster_t handle, const char* key);
int xgb_booster_get_attr(xgb_booster_t handle,
                         const char* key,
                         uint64_t buffer_capacity,
                         char* out_buffer,
                         uint64_t* out_len,
                         int* found);
int xgb_booster_get_attr_names(xgb_booster_t handle,
                               uint64_t buffer_capacity,
                               char* out_buffer,
                               uint64_t* out_len,
                               uint64_t* out_count);

/* Booster feature names/types. */
int xgb_booster_set_str_feature_info(xgb_booster_t handle,
                                     const char* field,
                                     const char** values,
                                     size_t len);
int xgb_booster_get_str_feature_info(xgb_booster_t handle,
                                     const char* field,
                                     uint64_t buffer_capacity,
                                     char* out_buffer,
                                     uint64_t* out_len,
                                     uint64_t* out_count);

/* Model dump APIs.  String arrays are copied into a NUL-separated buffer. */
int xgb_booster_dump_model(xgb_booster_t handle,
                           const char* format,
                           int with_stats,
                           uint64_t buffer_capacity,
                           char* out_buffer,
                           uint64_t* out_len,
                           uint64_t* out_count);

int xgb_booster_dump_model_with_features(xgb_booster_t handle,
                                         const char** feature_names,
                                         const char** feature_types,
                                         size_t n_features,
                                         const char* format,
                                         int with_stats,
                                         uint64_t buffer_capacity,
                                         char* out_buffer,
                                         uint64_t* out_len,
                                         uint64_t* out_count);

/* Feature score / importance.  Features are copied as NUL-separated strings;
 * shape and scores are copied into caller-owned arrays. */
int xgb_booster_feature_score(xgb_booster_t handle,
                              const char* config,
                              uint64_t feature_capacity,
                              char* out_features,
                              uint64_t* out_feature_len,
                              uint64_t* out_n_features,
                              uint64_t shape_capacity,
                              uint64_t* out_shape,
                              uint64_t* out_dim,
                              uint64_t score_capacity,
                              float* out_scores,
                              uint64_t* out_n_scores);

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
