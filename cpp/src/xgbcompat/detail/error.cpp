#include "xgbcompat/detail/error.hpp"

#include <xgboost/c_api.h>

#include <stdexcept>
#include <string>

namespace xgbcompat {

thread_local std::string g_last_error;

void check(int rc, const char* where) {
  if (rc != 0) {
    const char* msg = XGBGetLastError();
    g_last_error = std::string(where) + ": " + (msg ? msg : "<null>");
    throw std::runtime_error(g_last_error);
  }
}

std::string last_error() {
  return g_last_error;
}

}  // namespace xgbcompat
