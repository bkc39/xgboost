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

TEST(XgbCompatTest, RegressionDemoLearnsTrainingData) {
  const auto result = xgbcompat::run_regression_demo();
  ASSERT_EQ(result.predictions.size(), 8u);
  for (float p : result.predictions) {
    EXPECT_TRUE(std::isfinite(p));
  }
  // Labels from examples.cpp
  const std::vector<float> labels = {3.5f, 3.5f, 6.5f, 2.0f,
                                     9.0f, 4.0f, 7.0f, 1.0f};
  double sq_err = 0.0;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const double d = result.predictions[i] - labels[i];
    sq_err += d * d;
  }
  const double mse = sq_err / static_cast<double>(labels.size());
  // Loose threshold: training MSE should be well under variance of labels (~7).
  EXPECT_LT(mse, 3.0) << "regression failed to fit training data";
}

TEST(XgbCompatTest, ClassificationDemoProducesProbabilities) {
  const auto result = xgbcompat::run_classification_demo();
  ASSERT_EQ(result.predictions.size(), 10u);
  for (float p : result.predictions) {
    EXPECT_TRUE(std::isfinite(p));
    EXPECT_GE(p, 0.0f);
    EXPECT_LE(p, 1.0f);
  }
  // The two classes are linearly separable; check rough class separation.
  const std::vector<int> truth = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  int correct = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const int pred = result.predictions[i] > 0.5f ? 1 : 0;
    if (pred == truth[i])
      ++correct;
  }
  EXPECT_GE(correct, 9);
}

TEST(XgbCompatTest, ExternCRegressionReturnsZero) {
  double first = 0.0;
  const int rc = xgb_run_regression_demo(&first);
  EXPECT_EQ(rc, 0) << "last error: " << xgb_last_error();
  EXPECT_TRUE(std::isfinite(first));
}

TEST(XgbCompatTest, ExternCClassificationReturnsProbability) {
  double first = 0.0;
  const int rc = xgb_run_classification_demo(&first);
  EXPECT_EQ(rc, 0) << "last error: " << xgb_last_error();
  EXPECT_GE(first, 0.0);
  EXPECT_LE(first, 1.0);
}

}  // namespace
