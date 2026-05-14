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

int xgb_booster_set_param(xgb_booster_t handle, const char* key,
                          const char* value) {
  if (!handle || !key || !value) {
    xgbcompat::g_last_error = "xgb_booster_set_param: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterSetParam(static_cast<BoosterHandle>(handle), key, value),
        "XGBoosterSetParam");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_set_param: unknown exception";
    return 1;
  }
}

int xgb_booster_update_one_iter(xgb_booster_t handle, int iter,
                                xgb_dmatrix_t dtrain) {
  if (!handle || !dtrain) {
    xgbcompat::g_last_error = "xgb_booster_update_one_iter: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(
        XGBoosterUpdateOneIter(static_cast<BoosterHandle>(handle), iter,
                               static_cast<DMatrixHandle>(dtrain)),
        "XGBoosterUpdateOneIter");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_update_one_iter: unknown exception";
    return 1;
  }
}

int xgb_booster_train_one_iter(xgb_booster_t handle, xgb_dmatrix_t dtrain,
                               int iter, const char* grad, const char* hess) {
  if (!handle || !dtrain || !grad || !hess) {
    xgbcompat::g_last_error = "xgb_booster_train_one_iter: invalid argument";
    return 1;
  }
  try {
    xgbcompat::check(XGBoosterTrainOneIter(static_cast<BoosterHandle>(handle),
                                           static_cast<DMatrixHandle>(dtrain),
                                           iter, grad, hess),
                     "XGBoosterTrainOneIter");
    return 0;
  } catch (const std::exception&) {
    return 1;
  } catch (...) {
    xgbcompat::g_last_error = "xgb_booster_train_one_iter: unknown exception";
    return 1;
  }
}

}  // extern "C"
