#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xgbcompat/c_api/dmatrix.h"

#ifdef __cplusplus
extern "C" {
#endif

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
xgb_booster_t xgb_booster_slice(xgb_booster_t handle, int begin_layer,
                                int end_layer, int step);

/* Query booster structure. */
int xgb_booster_boosted_rounds(xgb_booster_t handle, int* out_rounds);
int xgb_booster_num_feature(xgb_booster_t handle, uint64_t* out_features);

/* Set a (string-valued) booster parameter, e.g. ("objective",
 * "reg:squarederror") or ("max_depth", "6"). */
int xgb_booster_set_param(xgb_booster_t handle, const char* key,
                          const char* value);

/* Run one boosting round on `dtrain` at iteration `iter`. */
int xgb_booster_update_one_iter(xgb_booster_t handle, int iter,
                                xgb_dmatrix_t dtrain);

/* Run one custom-objective boosting round using caller-provided gradient and
 * Hessian array-interface JSON strings. */
int xgb_booster_train_one_iter(xgb_booster_t handle, xgb_dmatrix_t dtrain,
                               int iter, const char* grad, const char* hess);

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
int xgb_booster_predict(xgb_booster_t handle, xgb_dmatrix_t dmat,
                        const char* config, uint64_t buffer_capacity,
                        float* out_buffer, uint64_t* out_len);

/* CPU inplace prediction from array-interface inputs.  These mirror
 * `xgb_booster_predict`: XGBoost-owned prediction memory is copied into
 * caller-owned buffers, and rc=2 reports the required float count. */
int xgb_booster_predict_from_dense(xgb_booster_t handle, const char* values,
                                   const char* config, xgb_dmatrix_t proxy,
                                   uint64_t buffer_capacity, float* out_buffer,
                                   uint64_t* out_len);
int xgb_booster_predict_from_csr(xgb_booster_t handle, const char* indptr,
                                 const char* indices, const char* values,
                                 uint64_t ncol, const char* config,
                                 xgb_dmatrix_t proxy, uint64_t buffer_capacity,
                                 float* out_buffer, uint64_t* out_len);
int xgb_booster_predict_from_columnar(xgb_booster_t handle, const char* values,
                                      const char* config, xgb_dmatrix_t proxy,
                                      uint64_t buffer_capacity,
                                      float* out_buffer, uint64_t* out_len);

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
int xgb_booster_save_model_to_buffer(xgb_booster_t handle, const char* config,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len);

/* Load a booster from a byte buffer produced by save_model_to_buffer (or
 * an externally provided XGBoost JSON/UBJ blob). */
int xgb_booster_load_model_from_buffer(xgb_booster_t handle, const void* data,
                                       uint64_t len);

/* Memory snapshot serialization.  Unlike save_model_to_buffer, this captures
 * the full booster state (training caches, iteration counters, etc.) so a
 * snapshot taken mid-training can resume `update_one_iter` calls in a fresh
 * booster.  Same size-then-fill contract as save_model_to_buffer. */
int xgb_booster_serialize_to_buffer(xgb_booster_t handle,
                                    uint64_t buffer_capacity, char* out_buffer,
                                    uint64_t* out_len);

/* Restore a booster from a snapshot produced by serialize_to_buffer. */
int xgb_booster_unserialize_from_buffer(xgb_booster_t handle, const void* data,
                                        uint64_t len);

/* Save/load XGBoost's internal JSON config. */
int xgb_booster_save_json_config(xgb_booster_t handle, uint64_t buffer_capacity,
                                 char* out_buffer, uint64_t* out_len);
int xgb_booster_load_json_config(xgb_booster_t handle, const char* config);

/* Booster attributes.  `get` uses a size-then-fill contract and reports
 * `found=0` for missing attrs. */
int xgb_booster_set_attr(xgb_booster_t handle, const char* key,
                         const char* value);
int xgb_booster_delete_attr(xgb_booster_t handle, const char* key);
int xgb_booster_get_attr(xgb_booster_t handle, const char* key,
                         uint64_t buffer_capacity, char* out_buffer,
                         uint64_t* out_len, int* found);
int xgb_booster_get_attr_names(xgb_booster_t handle, uint64_t buffer_capacity,
                               char* out_buffer, uint64_t* out_len,
                               uint64_t* out_count);

/* Booster feature names/types. */
int xgb_booster_set_str_feature_info(xgb_booster_t handle, const char* field,
                                     const char** values, size_t len);
int xgb_booster_get_str_feature_info(xgb_booster_t handle, const char* field,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len, uint64_t* out_count);

/* Model dump APIs.  String arrays are copied into a NUL-separated buffer. */
int xgb_booster_dump_model(xgb_booster_t handle, const char* format,
                           int with_stats, uint64_t buffer_capacity,
                           char* out_buffer, uint64_t* out_len,
                           uint64_t* out_count);

int xgb_booster_dump_model_with_features(
    xgb_booster_t handle, const char** feature_names,
    const char** feature_types, size_t n_features, const char* format,
    int with_stats, uint64_t buffer_capacity, char* out_buffer,
    uint64_t* out_len, uint64_t* out_count);

/* Feature score / importance.  Features are copied as NUL-separated strings;
 * shape and scores are copied into caller-owned arrays. */
int xgb_booster_feature_score(xgb_booster_t handle, const char* config,
                              uint64_t feature_capacity, char* out_features,
                              uint64_t* out_feature_len,
                              uint64_t* out_n_features, uint64_t shape_capacity,
                              uint64_t* out_shape, uint64_t* out_dim,
                              uint64_t score_capacity, float* out_scores,
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
int xgb_booster_eval_one_iter(xgb_booster_t handle, int iter,
                              const xgb_dmatrix_t* dmats, const char** names,
                              size_t n_dmats, uint64_t buffer_capacity,
                              char* out_buffer, uint64_t* out_len);

#ifdef __cplusplus
}
#endif
