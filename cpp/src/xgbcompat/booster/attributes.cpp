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

int xgb_booster_set_attr(xgb_booster_t handle, const char* key,
                         const char* value) {
  if (!handle || !key || !value) {
    xgbcompat::g_last_error = "xgb_booster_set_attr: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSetAttr(static_cast<BoosterHandle>(handle), key, value),
        "XGBoosterSetAttr");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_set_attr: unknown exception";
    return 1;
  }
}

int xgb_booster_delete_attr(xgb_booster_t handle, const char* key) {
  if (!handle || !key) {
    xgbcompat::g_last_error = "xgb_booster_delete_attr: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSetAttr(static_cast<BoosterHandle>(handle), key, nullptr),
        "XGBoosterSetAttr(delete)");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_delete_attr: unknown exception";
    return 1;
  }
}

int xgb_booster_get_attr(xgb_booster_t handle, const char* key,
                         uint64_t buffer_capacity, char* out_buffer,
                         uint64_t* out_len, int* found) {
  if (!handle || !key || !out_len || !found ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error = "xgb_booster_get_attr: invalid argument";
    return 1;
  }
  *out_len = 0;
  *found = 0;
  try {
    const char* value = nullptr;
    int success = 0;
    xgbcompat::check(XGBoosterGetAttr(static_cast<BoosterHandle>(handle), key,
                                      &value, &success),
                     "XGBoosterGetAttr");
    *found = success;
    if (!success || !value) {
      return 0;
    }
    const uint64_t len = static_cast<uint64_t>(std::strlen(value));
    *out_len = len;
    if (buffer_capacity < len) {
      return 2;
    }
    if (len > 0 && out_buffer) {
      std::copy(value, value + len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_get_attr: unknown exception";
    return 1;
  }
}

int xgb_booster_get_attr_names(xgb_booster_t handle, uint64_t buffer_capacity,
                               char* out_buffer, uint64_t* out_len,
                               uint64_t* out_count) {
  if (!handle || !out_len || !out_count ||
      (buffer_capacity > 0 && !out_buffer)) {
    xgbcompat::g_last_error = "xgb_booster_get_attr_names: invalid argument";
    return 1;
  }
  *out_len = 0;
  *out_count = 0;
  try {
    bst_ulong n = 0;
    const char** names = nullptr;
    xgbcompat::check(
        XGBoosterGetAttrNames(static_cast<BoosterHandle>(handle), &n, &names),
        "XGBoosterGetAttrNames");
    const uint64_t need = xgbcompat::nul_separated_size(names, n);
    *out_len = need;
    *out_count = static_cast<uint64_t>(n);
    if (buffer_capacity < need) {
      return 2;
    }
    if (need > 0 && out_buffer) {
      xgbcompat::copy_nul_separated(names, n, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_get_attr_names: unknown exception";
    return 1;
  }
}

}  // extern "C"
