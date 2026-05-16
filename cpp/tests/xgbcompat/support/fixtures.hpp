#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "xgbcompat/xgbcompat.hpp"

namespace xgbcompat_test_support {

inline std::string array_interface(std::uintptr_t ptr,
                                   const std::string& typestr,
                                   std::initializer_list<std::uint64_t> shape) {
  std::ostringstream os;
  os << "{\"data\":[" << ptr << ",false],\"typestr\":\"" << typestr
     << "\",\"shape\":[";
  bool first = true;
  for (std::uint64_t dim : shape) {
    if (!first)
      os << ",";
    first = false;
    os << dim;
  }
  os << "],\"version\":3}";
  return os.str();
}

// A small known-fittable regression target.
struct RegressionFixture {
  std::vector<float> features;
  std::vector<float> labels;
  std::uint64_t nrow;
  std::uint64_t ncol;
};

// XGBoost requires `type` in the predict config; "{}" triggers a fatal
// "Argument `type` is required".  This is the standard inference config.
constexpr const char* kPredictConfig =
    "{\"type\":0,\"training\":false,"
    "\"iteration_begin\":0,\"iteration_end\":0,"
    "\"strict_shape\":false}";

constexpr const char* kInplacePredictConfig =
    "{\"type\":0,\"training\":false,"
    "\"iteration_begin\":0,\"iteration_end\":0,"
    "\"strict_shape\":false,\"missing\":-1.0}";

inline RegressionFixture make_regression_fixture() {
  return {
      /*features=*/{
          1.0f, 2.0f, 0.5f, 2.0f, 1.0f, 1.5f, 3.0f, 0.5f,
          0.0f, 0.5f, 3.0f, 2.0f, 4.0f, 2.0f, 1.0f, 1.5f,
          1.5f, 0.5f, 2.5f, 3.5f, 1.5f, 0.0f, 1.0f, 0.0f,
      },
      /*labels=*/{3.5f, 3.5f, 6.5f, 2.0f, 9.0f, 4.0f, 7.0f, 1.0f},
      /*nrow=*/8,
      /*ncol=*/3,
  };
}

inline xgb_dmatrix_t make_train_matrix(const RegressionFixture& f) {
  xgb_dmatrix_t dtrain =
      xgb_dmatrix_create_from_mat(f.features.data(), f.nrow, f.ncol, -1.0f);
  if (dtrain) {
    xgb_dmatrix_set_float_info(dtrain, "label", f.labels.data(),
                               f.labels.size());
  }
  return dtrain;
}

inline std::vector<float> predict_with_dmatrix(xgb_booster_t booster,
                                               xgb_dmatrix_t dmat,
                                               std::size_t nrow) {
  std::vector<float> preds(nrow, 0.0f);
  std::uint64_t len = 0;
  xgb_booster_predict(booster, dmat, kPredictConfig, preds.size(), preds.data(),
                      &len);
  preds.resize(len);
  return preds;
}

// Train a small regressor for serialization and inspection tests.  Returns
// owned handles; callers must free both.
struct TrainedModel {
  xgb_booster_t booster;
  xgb_dmatrix_t dtrain;
  std::vector<float> baseline_predictions;
};

inline TrainedModel train_small_regressor(int rounds) {
  TrainedModel out{};
  const auto f = make_regression_fixture();
  out.dtrain = make_train_matrix(f);
  if (!out.dtrain)
    return out;
  const xgb_dmatrix_t cache[] = {out.dtrain};
  out.booster = xgb_booster_create(cache, 1);
  if (!out.booster)
    return out;
  xgb_booster_set_param(out.booster, "objective", "reg:squarederror");
  xgb_booster_set_param(out.booster, "max_depth", "3");
  xgb_booster_set_param(out.booster, "eta", "0.1");
  xgb_booster_set_param(out.booster, "verbosity", "0");
  for (int i = 0; i < rounds; ++i) {
    xgb_booster_update_one_iter(out.booster, i, out.dtrain);
  }
  out.baseline_predictions.assign(f.nrow, 0.0f);
  std::uint64_t len = 0;
  xgb_booster_predict(out.booster, out.dtrain, kPredictConfig,
                      out.baseline_predictions.size(),
                      out.baseline_predictions.data(), &len);
  return out;
}

inline std::string make_temp_path(const char* suffix) {
  return ::testing::TempDir() + "xgbcompat_model_" +
         std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
         suffix;
}

}  // namespace xgbcompat_test_support
