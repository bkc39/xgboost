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

TEST(XgbBoosterTest, CreateFreeRoundTrip) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr) << "last error: " << xgb_last_error();
  xgb_booster_free(b);
}

TEST(XgbBoosterTest, CreateWithCacheRoundTrip) {
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr) << "last error: " << xgb_last_error();
  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

TEST(XgbBoosterTest, CacheLenWithoutPointerIsError) {
  xgb_booster_t b = xgb_booster_create(nullptr, 1);
  EXPECT_EQ(b, nullptr);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
}

TEST(XgbBoosterTest, SetParamSuccess) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  EXPECT_EQ(xgb_booster_set_param(b, "max_depth", "3"), 0);
  EXPECT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);
  xgb_booster_free(b);
}

TEST(XgbBoosterTest, LifecycleQueriesResetAndSlice) {
  TrainedModel m = train_small_regressor(8);
  ASSERT_NE(m.booster, nullptr);
  ASSERT_NE(m.dtrain, nullptr);

  int rounds = 0;
  ASSERT_EQ(xgb_booster_boosted_rounds(m.booster, &rounds), 0)
      << "last error: " << xgb_last_error();
  EXPECT_EQ(rounds, 8);

  std::uint64_t n_features = 0;
  ASSERT_EQ(xgb_booster_num_feature(m.booster, &n_features), 0)
      << "last error: " << xgb_last_error();
  EXPECT_EQ(n_features, 3u);

  xgb_booster_t sliced = xgb_booster_slice(m.booster, 0, 3, 1);
  ASSERT_NE(sliced, nullptr) << "last error: " << xgb_last_error();
  int sliced_rounds = 0;
  ASSERT_EQ(xgb_booster_boosted_rounds(sliced, &sliced_rounds), 0)
      << "last error: " << xgb_last_error();
  EXPECT_EQ(sliced_rounds, 3);

  ASSERT_EQ(xgb_booster_reset(m.booster), 0)
      << "last error: " << xgb_last_error();

  xgb_booster_free(sliced);
  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

}  // namespace
