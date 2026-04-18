#include "xgbcompat/xgbcompat.hpp"

#include <xgboost/c_api.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace xgbcompat {

namespace {

thread_local std::string g_last_error;

void check(int rc, const char* where) {
  if (rc != 0) {
    const char* msg = XGBGetLastError();
    g_last_error = std::string(where) + ": " + (msg ? msg : "<null>");
    throw std::runtime_error(g_last_error);
  }
}

DemoResult train_and_predict(const std::vector<float>& features,
                             const std::vector<float>& labels,
                             std::uint64_t nrow,
                             std::uint64_t ncol,
                             const char* objective,
                             int num_round) {
  DMatrixHandle dtrain = nullptr;
  BoosterHandle booster = nullptr;

  try {
    check(XGDMatrixCreateFromMat(features.data(),
                                 static_cast<bst_ulong>(nrow),
                                 static_cast<bst_ulong>(ncol),
                                 /*missing=*/-1.0f,
                                 &dtrain),
          "XGDMatrixCreateFromMat");

    check(XGDMatrixSetFloatInfo(dtrain, "label",
                                labels.data(),
                                static_cast<bst_ulong>(labels.size())),
          "XGDMatrixSetFloatInfo(label)");

    const DMatrixHandle cache[] = {dtrain};
    check(XGBoosterCreate(cache, 1, &booster), "XGBoosterCreate");

    check(XGBoosterSetParam(booster, "objective", objective),
          "XGBoosterSetParam(objective)");
    check(XGBoosterSetParam(booster, "max_depth", "3"),
          "XGBoosterSetParam(max_depth)");
    check(XGBoosterSetParam(booster, "eta", "0.1"),
          "XGBoosterSetParam(eta)");
    check(XGBoosterSetParam(booster, "verbosity", "0"),
          "XGBoosterSetParam(verbosity)");

    for (int iter = 0; iter < num_round; ++iter) {
      check(XGBoosterUpdateOneIter(booster, iter, dtrain),
            "XGBoosterUpdateOneIter");
    }

    const char* config =
        "{\"type\": 0, \"training\": false, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, "
        "\"strict_shape\": false}";
    bst_ulong const* out_shape = nullptr;
    bst_ulong out_dim = 0;
    float const* out_result = nullptr;
    check(XGBoosterPredictFromDMatrix(booster, dtrain, config,
                                      &out_shape, &out_dim, &out_result),
          "XGBoosterPredictFromDMatrix");

    std::uint64_t n_out = 1;
    for (bst_ulong i = 0; i < out_dim; ++i) {
      n_out *= out_shape[i];
    }

    DemoResult result;
    result.predictions.assign(out_result, out_result + n_out);

    XGBoosterFree(booster);
    XGDMatrixFree(dtrain);
    return result;
  } catch (...) {
    if (booster) XGBoosterFree(booster);
    if (dtrain) XGDMatrixFree(dtrain);
    throw;
  }
}

}  // namespace

std::string version() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  XGBoostVersion(&major, &minor, &patch);
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
}

std::string last_error() {
  return g_last_error;
}

DemoResult run_regression_demo() {
  // 8 rows x 3 features; label ~ 2*x0 + x1 - x2
  static constexpr std::uint64_t nrow = 8;
  static constexpr std::uint64_t ncol = 3;
  const std::vector<float> features = {
      1.0f, 2.0f, 0.5f,
      2.0f, 1.0f, 1.5f,
      3.0f, 0.5f, 0.0f,
      0.5f, 3.0f, 2.0f,
      4.0f, 2.0f, 1.0f,
      1.5f, 1.5f, 0.5f,
      2.5f, 3.5f, 1.5f,
      0.0f, 1.0f, 0.0f,
  };
  const std::vector<float> labels = {3.5f, 3.5f, 6.5f, 2.0f, 9.0f,
                                     4.0f, 7.0f, 1.0f};
  return train_and_predict(features, labels, nrow, ncol,
                           "reg:squarederror", 50);
}

DemoResult run_classification_demo() {
  // 10 rows x 4 features; binary label based on sum(features) > threshold
  static constexpr std::uint64_t nrow = 10;
  static constexpr std::uint64_t ncol = 4;
  const std::vector<float> features = {
      0.1f, 0.2f, 0.1f, 0.0f,
      5.0f, 4.0f, 5.5f, 6.0f,
      0.3f, 0.5f, 0.1f, 0.2f,
      4.5f, 5.0f, 4.0f, 5.5f,
      0.0f, 0.1f, 0.2f, 0.0f,
      6.0f, 5.5f, 6.5f, 5.0f,
      0.4f, 0.3f, 0.2f, 0.5f,
      5.5f, 6.0f, 4.5f, 5.0f,
      0.2f, 0.1f, 0.3f, 0.1f,
      4.0f, 4.5f, 5.0f, 4.0f,
  };
  const std::vector<float> labels = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  return train_and_predict(features, labels, nrow, ncol,
                           "binary:logistic", 30);
}

}  // namespace xgbcompat
