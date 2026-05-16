#pragma once

#include <string>

namespace xgbcompat {

// Thread-local buffer surfaced through xgb_last_error().
extern thread_local std::string g_last_error;

void check(int rc, const char* where);
std::string last_error();

}  // namespace xgbcompat
