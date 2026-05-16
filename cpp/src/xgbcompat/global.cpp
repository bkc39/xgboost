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

namespace xgbcompat {
namespace {
thread_local std::string g_version_cache;
}
}  // namespace xgbcompat

namespace xgbcompat {

std::string version() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  XGBoostVersion(&major, &minor, &patch);
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
}

}  // namespace xgbcompat

extern "C" {

const char* xgb_version(void) {
  try {
    xgbcompat::g_version_cache = xgbcompat::version();
    return xgbcompat::g_version_cache.c_str();
  } catch (...) {
    return "";
  }
}

const char* xgb_last_error(void) {
  static thread_local std::string buf;
  buf = xgbcompat::last_error();
  return buf.c_str();
}

int xgb_build_info(uint64_t buffer_capacity, char* out_buffer,
                   uint64_t* out_len) {
  if (!out_len) {
    xgbcompat::g_last_error = "xgb_build_info: out_len is null";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_build_info: capacity > 0 but out_buffer is null";
    return 1;
  }
  *out_len = 0;
  try {
    const char* info = nullptr;
    xgbcompat::check(XGBuildInfo(&info), "XGBuildInfo");
    const size_t len = info ? std::strlen(info) : 0;
    *out_len = static_cast<uint64_t>(len);
    if (buffer_capacity < *out_len) {
      return 2;
    }
    if (len > 0 && out_buffer) {
      std::copy(info, info + len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_build_info: unknown exception";
    return 1;
  }
}

int xgb_get_global_config(uint64_t buffer_capacity, char* out_buffer,
                          uint64_t* out_len) {
  if (!out_len) {
    xgbcompat::g_last_error = "xgb_get_global_config: out_len is null";
    return 1;
  }
  if (buffer_capacity > 0 && !out_buffer) {
    xgbcompat::g_last_error =
        "xgb_get_global_config: capacity > 0 but out_buffer is null";
    return 1;
  }
  *out_len = 0;
  try {
    const char* config = nullptr;
    xgbcompat::check(XGBGetGlobalConfig(&config), "XGBGetGlobalConfig");
    const size_t len = config ? std::strlen(config) : 0;
    *out_len = static_cast<uint64_t>(len);
    if (buffer_capacity < *out_len) {
      return 2;
    }
    if (len > 0 && out_buffer) {
      std::copy(config, config + len, out_buffer);
    }
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_get_global_config: unknown exception";
    return 1;
  }
}

int xgb_set_global_config(const char* config) {
  if (!config) {
    xgbcompat::g_last_error = "xgb_set_global_config: config is null";
    return 1;
  }
  try {
    xgbcompat::check(XGBSetGlobalConfig(config), "XGBSetGlobalConfig");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_set_global_config: unknown exception";
    return 1;
  }
}

int xgb_register_log_callback(void (*callback)(const char*)) {
  if (!callback) {
    xgbcompat::g_last_error = "xgb_register_log_callback: callback is null";
    return 1;
  }
  try {
    xgbcompat::check(XGBRegisterLogCallback(callback),
                     "XGBRegisterLogCallback");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_register_log_callback: unknown exception";
    return 1;
  }
}

}  // extern "C"
