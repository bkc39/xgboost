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

int xgb_booster_save_model(xgb_booster_t handle, const char* path) {
  if (!handle || !path) {
    xgbcompat::g_last_error = "xgb_booster_save_model: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSaveModel(static_cast<BoosterHandle>(handle), path),
        "XGBoosterSaveModel");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_save_model: unknown exception";
    return 1;
  }
}

int xgb_booster_load_model(xgb_booster_t handle, const char* path) {
  if (!handle || !path) {
    xgbcompat::g_last_error = "xgb_booster_load_model: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterLoadModel(static_cast<BoosterHandle>(handle), path),
        "XGBoosterLoadModel");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_load_model: unknown exception";
    return 1;
  }
}

int xgb_booster_save_model_to_buffer(xgb_booster_t handle, const char* config,
                                     uint64_t buffer_capacity, char* out_buffer,
                                     uint64_t* out_len) {
  if (!handle || !config || !out_len) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: invalid argument";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: capacity > 0 but out_buffer null";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong n = 0;
    const char* data = nullptr;
    xgbcompat::check(XGBoosterSaveModelToBuffer(
                         static_cast<BoosterHandle>(handle), config, &n, &data),
                     "XGBoosterSaveModelToBuffer");
    *out_len = static_cast<uint64_t>(n);
    if (buffer_capacity < *out_len) {
      // Booster-owned buffer is invalidated on the next call; do not leak.
      return 2;
    }
    if (*out_len > 0 && data && out_buffer) {
      std::copy(data, data + *out_len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_save_model_to_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_load_model_from_buffer(xgb_booster_t handle, const void* data,
                                       uint64_t len) {
  if (!handle || (!data && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_booster_load_model_from_buffer: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterLoadModelFromBuffer(static_cast<BoosterHandle>(handle), data,
                                     static_cast<bst_ulong>(len)),
        "XGBoosterLoadModelFromBuffer");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_load_model_from_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_serialize_to_buffer(xgb_booster_t handle,
                                    uint64_t buffer_capacity, char* out_buffer,
                                    uint64_t* out_len) {
  if (!handle || !out_len || (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error =
        "xgb_booster_serialize_to_buffer: invalid argument";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong n = 0;
    const char* data = nullptr;
    xgbcompat::check(XGBoosterSerializeToBuffer(
                         static_cast<BoosterHandle>(handle), &n, &data),
                     "XGBoosterSerializeToBuffer");
    *out_len = static_cast<uint64_t>(n);
    if (buffer_capacity < *out_len) {
      // Booster-owned buffer is invalidated by the next serialization call;
      // copy on every successful return so it never escapes.
      return 2;
    }
    if (*out_len > 0 && data && out_buffer) {
      std::copy(data, data + *out_len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_serialize_to_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_unserialize_from_buffer(xgb_booster_t handle, const void* data,
                                        uint64_t len) {
  if (!handle || (!data && len > 0)) {
    xgbcompat::g_last_error =
        "xgb_booster_unserialize_from_buffer: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterUnserializeFromBuffer(static_cast<BoosterHandle>(handle), data,
                                       static_cast<bst_ulong>(len)),
        "XGBoosterUnserializeFromBuffer");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error =
        "xgb_booster_unserialize_from_buffer: unknown exception";
    return 1;
  }
}

int xgb_booster_save_json_config(xgb_booster_t handle, uint64_t buffer_capacity,
                                 char* out_buffer, uint64_t* out_len) {
  if (!handle || !out_len || (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error = "xgb_booster_save_json_config: invalid argument";
    return 1;
  }
  *out_len = 0;
  try {
    bst_ulong n = 0;
    const char* config = nullptr;
    xgbcompat::check(XGBoosterSaveJsonConfig(static_cast<BoosterHandle>(handle),
                                             &n, &config),
                     "XGBoosterSaveJsonConfig");
    *out_len = static_cast<uint64_t>(n);
    if (buffer_capacity < *out_len) {
      return 2;
    }
    if (*out_len > 0 && config && out_buffer) {
      std::copy(config, config + *out_len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_save_json_config: unknown exception";
    return 1;
  }
}

int xgb_booster_load_json_config(xgb_booster_t handle, const char* config) {
  if (!handle || !config) {
    xgbcompat::g_last_error = "xgb_booster_load_json_config: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterLoadJsonConfig(static_cast<BoosterHandle>(handle), config),
        "XGBoosterLoadJsonConfig");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_load_json_config: unknown exception";
    return 1;
  }
}

}  // extern "C"
