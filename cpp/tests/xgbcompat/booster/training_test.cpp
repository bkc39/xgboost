#include <gtest/gtest.h>
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "xgbcompat/support/fixtures.hpp"
#include "xgbcompat/xgbcompat.hpp"

namespace {

using namespace xgbcompat_test_support;

TEST(XgbBoosterTest, TrainOneIterWithCustomSquaredErrorObjective) {
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr) << "last error: " << xgb_last_error();
  ASSERT_EQ(xgb_booster_set_param(b, "max_depth", "3"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "eta", "0.2"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);

  std::vector<float> initial = predict_with_dmatrix(b, dtrain, f.nrow);
  const auto mse = [&](const std::vector<float>& preds) {
    double total = 0.0;
    for (std::size_t i = 0; i < f.labels.size(); ++i) {
      const double d = preds[i] - f.labels[i];
      total += d * d;
    }
    return total / static_cast<double>(f.labels.size());
  };
  const double initial_mse = mse(initial);

  std::vector<float> grad(f.nrow, 0.0f);
  std::vector<float> hess(f.nrow, 1.0f);
  for (int iter = 0; iter < 20; ++iter) {
    const std::vector<float> preds = predict_with_dmatrix(b, dtrain, f.nrow);
    for (std::size_t i = 0; i < f.nrow; ++i) {
      grad[i] = preds[i] - f.labels[i];
      hess[i] = 1.0f;
    }
    const std::string grad_ai = array_interface(
        reinterpret_cast<std::uintptr_t>(grad.data()), "<f4", {grad.size()});
    const std::string hess_ai = array_interface(
        reinterpret_cast<std::uintptr_t>(hess.data()), "<f4", {hess.size()});
    ASSERT_EQ(xgb_booster_train_one_iter(b, dtrain, iter, grad_ai.c_str(),
                                         hess_ai.c_str()),
              0)
        << "last error: " << xgb_last_error();
  }

  const std::vector<float> trained = predict_with_dmatrix(b, dtrain, f.nrow);
  const double trained_mse = mse(trained);
  EXPECT_LT(trained_mse, initial_mse);
  EXPECT_LT(trained_mse, 3.0);

  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

}  // namespace
