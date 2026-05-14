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

TEST(XgbBoosterTest, TrainAndPredictFitsTrainingData) {
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr);

  ASSERT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "max_depth", "3"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "eta", "0.1"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);

  for (int iter = 0; iter < 50; ++iter) {
    ASSERT_EQ(xgb_booster_update_one_iter(b, iter, dtrain), 0)
        << "iter " << iter << " err=" << xgb_last_error();
  }

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t out_len = 0;
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, preds.size(),
                                preds.data(), &out_len),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(out_len, f.nrow);

  double sq_err = 0.0;
  for (std::size_t i = 0; i < f.labels.size(); ++i) {
    const double d = preds[i] - f.labels[i];
    sq_err += d * d;
  }
  const double mse = sq_err / static_cast<double>(f.labels.size());
  EXPECT_LT(mse, 3.0) << "training MSE too high: " << mse;

  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

TEST(XgbBoosterTest, PredictTooSmallBufferReturnsRequiredSize) {
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr) << "last error: " << xgb_last_error();
  ASSERT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);
  ASSERT_EQ(xgb_booster_update_one_iter(b, 0, dtrain), 0)
      << "last error: " << xgb_last_error();

  // Capacity 0, NULL buffer: pure size probe.
  std::uint64_t out_len = 0;
  EXPECT_EQ(
      xgb_booster_predict(b, dtrain, kPredictConfig, 0, nullptr, &out_len), 2);
  EXPECT_EQ(out_len, f.nrow);

  // Capacity smaller than required: still rc=2, no copy.
  std::vector<float> small(f.nrow - 1, -42.0f);
  out_len = 0;
  EXPECT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, small.size(),
                                small.data(), &out_len),
            2);
  EXPECT_EQ(out_len, f.nrow);
  for (float v : small)
    EXPECT_FLOAT_EQ(v, -42.0f);

  // Resize and retry — should succeed now.
  std::vector<float> full(out_len, 0.0f);
  EXPECT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, full.size(),
                                full.data(), &out_len),
            0);
  EXPECT_EQ(out_len, f.nrow);

  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

TEST(XgbBoosterTest, PredictTwiceProducesIdenticalResults) {
  // Sanity-check the copy: with a deterministic model, calling predict twice
  // must yield byte-identical floats.  If the shim leaked the booster-owned
  // buffer, the second call could perturb the first result.
  const auto f = make_regression_fixture();
  xgb_dmatrix_t dtrain = make_train_matrix(f);
  ASSERT_NE(dtrain, nullptr);
  const xgb_dmatrix_t cache[] = {dtrain};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);
  for (int iter = 0; iter < 10; ++iter) {
    ASSERT_EQ(xgb_booster_update_one_iter(b, iter, dtrain), 0);
  }

  std::vector<float> first(f.nrow, 0.0f);
  std::vector<float> second(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, first.size(),
                                first.data(), &len),
            0);
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, second.size(),
                                second.data(), &len),
            0);
  for (std::size_t i = 0; i < f.nrow; ++i) {
    EXPECT_FLOAT_EQ(first[i], second[i]) << "i=" << i;
  }

  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

TEST(XgbBoosterTest, PredictFromDenseMatchesDMatrix) {
  TrainedModel m = train_small_regressor(12);
  ASSERT_NE(m.booster, nullptr);
  ASSERT_NE(m.dtrain, nullptr);
  const auto f = make_regression_fixture();
  const std::string data_ai =
      array_interface(reinterpret_cast<std::uintptr_t>(f.features.data()),
                      "<f4", {f.nrow, f.ncol});

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict_from_dense(m.booster, data_ai.c_str(),
                                           kInplacePredictConfig, nullptr,
                                           preds.size(), preds.data(), &len),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(len, f.nrow);
  for (std::size_t i = 0; i < f.nrow; ++i) {
    EXPECT_FLOAT_EQ(preds[i], m.baseline_predictions[i]) << "i=" << i;
  }

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterTest, PredictFromCSRMatchesDMatrix) {
  TrainedModel m = train_small_regressor(12);
  ASSERT_NE(m.booster, nullptr);
  const auto f = make_regression_fixture();
  std::vector<std::uint64_t> indptr(f.nrow + 1, 0);
  std::vector<std::uint32_t> indices(f.nrow * f.ncol, 0);
  for (std::size_t r = 0; r <= f.nrow; ++r) {
    indptr[r] = r * f.ncol;
  }
  for (std::size_t r = 0; r < f.nrow; ++r) {
    for (std::size_t c = 0; c < f.ncol; ++c) {
      indices[r * f.ncol + c] = static_cast<std::uint32_t>(c);
    }
  }
  const std::string indptr_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indptr.data()), "<u8", {indptr.size()});
  const std::string indices_ai =
      array_interface(reinterpret_cast<std::uintptr_t>(indices.data()), "<u4",
                      {indices.size()});
  const std::string data_ai =
      array_interface(reinterpret_cast<std::uintptr_t>(f.features.data()),
                      "<f4", {f.features.size()});

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict_from_csr(m.booster, indptr_ai.c_str(),
                                         indices_ai.c_str(), data_ai.c_str(),
                                         f.ncol, kInplacePredictConfig, nullptr,
                                         preds.size(), preds.data(), &len),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(len, f.nrow);
  for (std::size_t i = 0; i < f.nrow; ++i) {
    EXPECT_FLOAT_EQ(preds[i], m.baseline_predictions[i]) << "i=" << i;
  }

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterTest, PredictFromColumnarMatchesDMatrix) {
  TrainedModel m = train_small_regressor(12);
  ASSERT_NE(m.booster, nullptr);
  const auto f = make_regression_fixture();
  std::vector<float> col0(f.nrow, 0.0f);
  std::vector<float> col1(f.nrow, 0.0f);
  std::vector<float> col2(f.nrow, 0.0f);
  for (std::size_t r = 0; r < f.nrow; ++r) {
    col0[r] = f.features[r * f.ncol + 0];
    col1[r] = f.features[r * f.ncol + 1];
    col2[r] = f.features[r * f.ncol + 2];
  }
  const std::string col0_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col0.data()), "<f4", {f.nrow});
  const std::string col1_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col1.data()), "<f4", {f.nrow});
  const std::string col2_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col2.data()), "<f4", {f.nrow});
  const std::string data = "[" + col0_ai + "," + col1_ai + "," + col2_ai + "]";

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict_from_columnar(m.booster, data.c_str(),
                                              kInplacePredictConfig, nullptr,
                                              preds.size(), preds.data(), &len),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(len, f.nrow);
  for (std::size_t i = 0; i < f.nrow; ++i) {
    EXPECT_FLOAT_EQ(preds[i], m.baseline_predictions[i]) << "i=" << i;
  }

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

}  // namespace
