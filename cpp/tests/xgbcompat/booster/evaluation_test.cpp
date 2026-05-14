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

TEST(XgbBoosterEvalTest, ReturnsLineContainingProvidedNames) {
  TrainedModel m = train_small_regressor(5);
  ASSERT_NE(m.booster, nullptr);

  const xgb_dmatrix_t dmats[] = {m.dtrain};
  const char* names[] = {"train"};
  std::vector<char> buf(512, '\0');
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_eval_one_iter(m.booster, /*iter=*/4, dmats, names, 1,
                                      buf.size(), buf.data(), &len),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_GT(len, 0u);
  const std::string line(buf.data(), len);
  // Format: "[<iter>]\t<name>-<metric>:<value>"
  EXPECT_NE(line.find("[4]"), std::string::npos) << line;
  EXPECT_NE(line.find("train-"), std::string::npos) << line;
  EXPECT_NE(line.find(":"), std::string::npos) << line;

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterEvalTest, TooSmallBufferReportsRequiredSize) {
  TrainedModel m = train_small_regressor(2);
  ASSERT_NE(m.booster, nullptr);

  const xgb_dmatrix_t dmats[] = {m.dtrain};
  const char* names[] = {"train"};

  // Size probe: capacity 0, NULL buffer.
  std::uint64_t need = 0;
  EXPECT_EQ(xgb_booster_eval_one_iter(m.booster, 0, dmats, names, 1, 0, nullptr,
                                      &need),
            2);
  EXPECT_GT(need, 0u);

  // Capacity smaller than required: still rc=2, no copy.
  std::vector<char> small(need - 1, '\0');
  std::uint64_t need2 = 0;
  EXPECT_EQ(xgb_booster_eval_one_iter(m.booster, 0, dmats, names, 1,
                                      small.size(), small.data(), &need2),
            2);
  EXPECT_EQ(need2, need);

  // Resize and succeed.
  std::vector<char> full(need, '\0');
  std::uint64_t got = 0;
  EXPECT_EQ(xgb_booster_eval_one_iter(m.booster, 0, dmats, names, 1,
                                      full.size(), full.data(), &got),
            0);
  EXPECT_EQ(got, need);

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterEvalTest, MultipleDMatricesProduceMultipleMetrics) {
  // Two different DMatrices labeled "train" / "eval"; the result line must
  // contain both name prefixes.  We use the same fixture twice for
  // simplicity — the labels are what matters here.
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  xgb_dmatrix_t deval = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  ASSERT_NE(deval, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(xgb_booster_update_one_iter(b, i, dtrain), 0);
  }

  const xgb_dmatrix_t dmats[] = {dtrain, deval};
  const char* names[] = {"train", "eval"};
  std::vector<char> buf(1024, '\0');
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_eval_one_iter(b, 2, dmats, names, 2, buf.size(),
                                      buf.data(), &len),
            0);
  const std::string line(buf.data(), len);
  EXPECT_NE(line.find("train-"), std::string::npos) << line;
  EXPECT_NE(line.find("eval-"), std::string::npos) << line;

  xgb_booster_free(b);
  xgb_dmatrix_free(deval);
  xgb_dmatrix_free(dtrain);
}

}  // namespace
