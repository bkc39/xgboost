#pragma once

#include <xgboost/c_api.h>

#include "xgbcompat/c_api.h"

namespace xgbcompat::detail {

inline DMatrixHandle as_dmatrix(xgb_dmatrix_t handle) {
  return static_cast<DMatrixHandle>(handle);
}

inline BoosterHandle as_booster(xgb_booster_t handle) {
  return static_cast<BoosterHandle>(handle);
}

static_assert(sizeof(DMatrixHandle) == sizeof(xgb_dmatrix_t),
              "DMatrix handles must be ABI-compatible");
static_assert(sizeof(BoosterHandle) == sizeof(xgb_booster_t),
              "Booster handles must be ABI-compatible");

}  // namespace xgbcompat::detail
