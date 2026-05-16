#pragma once

#include <string>

#include "xgbcompat/c_api.h"

namespace xgbcompat {

// Underlying XGBoost library version as "major.minor.patch".
std::string version();

}  // namespace xgbcompat
