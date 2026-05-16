#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* xgb_version(void);

const char* xgb_last_error(void);

/* Return XGBoost build information as a UTF-8 JSON string.
 *
 * Same size-then-fill contract as booster prediction/string APIs:
 *   0: success, `*out_len` bytes written into `out_buffer`
 *   1: error — call `xgb_last_error()`
 *   2: out_buffer too small; `*out_len` holds the required size */
int xgb_build_info(uint64_t buffer_capacity, char* out_buffer,
                   uint64_t* out_len);

/* Read and write XGBoost's process-global JSON configuration. */
int xgb_get_global_config(uint64_t buffer_capacity, char* out_buffer,
                          uint64_t* out_len);
int xgb_set_global_config(const char* config);

/* Register a process-global XGBoost log callback.  The caller owns the
 * function pointer lifetime and must keep it alive for as long as XGBoost may
 * invoke it. */
int xgb_register_log_callback(void (*callback)(const char*));

#ifdef __cplusplus
}
#endif
