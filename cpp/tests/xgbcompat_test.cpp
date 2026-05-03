#include "xgbcompat/xgbcompat.hpp"

#include <gtest/gtest.h>

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace {

TEST(XgbCompatTest, VersionLooksLikeSemver) {
  const std::string v = xgbcompat::version();
  EXPECT_FALSE(v.empty());
  const std::regex semver(R"(^\d+\.\d+\.\d+$)");
  EXPECT_TRUE(std::regex_match(v, semver)) << "got: " << v;
}

TEST(XgbCompatTest, RegressionDemoLearnsTrainingData) {
  const auto result = xgbcompat::run_regression_demo();
  ASSERT_EQ(result.predictions.size(), 8u);
  for (float p : result.predictions) {
    EXPECT_TRUE(std::isfinite(p));
  }
  // Labels from examples.cpp
  const std::vector<float> labels = {3.5f, 3.5f, 6.5f, 2.0f, 9.0f,
                                     4.0f, 7.0f, 1.0f};
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
    if (pred == truth[i]) ++correct;
  }
  EXPECT_GE(correct, 9);
}

TEST(XgbCompatTest, ExternCVersionMatchesCpp) {
  const std::string cpp_v = xgbcompat::version();
  const char* c_v = xgb_version();
  ASSERT_NE(c_v, nullptr);
  EXPECT_EQ(cpp_v, std::string(c_v));
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

TEST(XgbCompatTest, BuildInfoReturnsJson) {
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_build_info(0, nullptr, &len), 2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(len, 0u);
  std::string info(len, '\0');
  ASSERT_EQ(xgb_build_info(info.size(), info.data(), &len), 0)
      << "last error: " << xgb_last_error();
  EXPECT_FALSE(info.empty());
  EXPECT_EQ(info.front(), '{');
}

TEST(XgbCompatTest, GlobalConfigRoundTrip) {
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_get_global_config(0, nullptr, &len), 2)
      << "last error: " << xgb_last_error();
  std::string before(len, '\0');
  ASSERT_EQ(xgb_get_global_config(before.size(), before.data(), &len), 0)
      << "last error: " << xgb_last_error();

  ASSERT_EQ(xgb_set_global_config("{\"verbosity\":0}"), 0)
      << "last error: " << xgb_last_error();
  len = 0;
  ASSERT_EQ(xgb_get_global_config(0, nullptr, &len), 2)
      << "last error: " << xgb_last_error();
  std::string after(len, '\0');
  ASSERT_EQ(xgb_get_global_config(after.size(), after.data(), &len), 0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(after.find("\"verbosity\":0"), std::string::npos);

  ASSERT_EQ(xgb_set_global_config(before.c_str()), 0)
      << "last error: " << xgb_last_error();
}

TEST(XgbCompatTest, SetGlobalConfigRejectsBadJson) {
  EXPECT_NE(xgb_set_global_config("{not-json"), 0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
}

TEST(XgbDMatrixTest, CreateFreeRoundTrip) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 4, 3, -1.0f);
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, CreateFromNullDataReturnsNull) {
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(nullptr, 4, 3, -1.0f);
  EXPECT_EQ(h, nullptr);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
}

TEST(XgbDMatrixTest, SetFloatInfoSuccess) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> labels = {0.0f, 1.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const int rc = xgb_dmatrix_set_float_info(h, "label", labels.data(),
                                            labels.size());
  EXPECT_EQ(rc, 0) << "last error: " << xgb_last_error();
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, SetFloatInfoBadField) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> vals = {0.0f, 1.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const int rc = xgb_dmatrix_set_float_info(h, "definitely_not_a_field",
                                            vals.data(), vals.size());
  EXPECT_NE(rc, 0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, NumRowNumCol) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 4, 3, -1.0f);
  ASSERT_NE(h, nullptr);
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 4u);
  EXPECT_EQ(ncol, 3u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, GetFloatInfoRoundTrip) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> labels = {0.5f, 1.5f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  ASSERT_EQ(xgb_dmatrix_set_float_info(h, "label", labels.data(),
                                       labels.size()),
            0);
  std::uint64_t len = 0;
  const float* ptr = nullptr;
  ASSERT_EQ(xgb_dmatrix_get_float_info(h, "label", &len, &ptr), 0);
  ASSERT_EQ(len, 2u);
  ASSERT_NE(ptr, nullptr);
  EXPECT_FLOAT_EQ(ptr[0], 0.5f);
  EXPECT_FLOAT_EQ(ptr[1], 1.5f);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, GetFloatInfoUnsetFieldIsEmpty) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  std::uint64_t len = 42;
  const float* ptr = reinterpret_cast<const float*>(0xdeadbeef);
  // "weight" was never set — expect success with len=0.
  ASSERT_EQ(xgb_dmatrix_get_float_info(h, "weight", &len, &ptr), 0);
  EXPECT_EQ(len, 0u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, NumNonMissingAndCsrRoundTrip) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  // Use a missing sentinel that is absent from `data` so every entry is
  // preserved through the CSR round-trip.
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 3, -1.0f);
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();

  std::uint64_t nnz = 0;
  ASSERT_EQ(xgb_dmatrix_num_non_missing(h, &nnz), 0);
  EXPECT_EQ(nnz, 6u);

  std::vector<std::uint64_t> indptr(3, 0);
  std::vector<std::uint32_t> indices(nnz, 0);
  std::vector<float> out_data(nnz, 0.0f);
  ASSERT_EQ(xgb_dmatrix_get_data_as_csr(h, "{}", indptr.data(),
                                        indices.data(), out_data.data()),
            0)
      << "last error: " << xgb_last_error();

  EXPECT_EQ(indptr[0], 0u);
  EXPECT_EQ(indptr[1], 3u);
  EXPECT_EQ(indptr[2], 6u);
  for (std::size_t i = 0; i < nnz; ++i) {
    EXPECT_EQ(indices[i], i % 3);
    EXPECT_FLOAT_EQ(out_data[i], data[i]);
  }
  xgb_dmatrix_free(h);
}

// ---------------------------------------------------------------------------
// Booster tests
// ---------------------------------------------------------------------------

namespace {

// Same data shape as run_regression_demo() so we have a known-fittable target.
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

RegressionFixture make_regression_fixture() {
  return {
      /*features=*/{
          1.0f, 2.0f, 0.5f, 2.0f, 1.0f, 1.5f, 3.0f, 0.5f, 0.0f,
          0.5f, 3.0f, 2.0f, 4.0f, 2.0f, 1.0f, 1.5f, 1.5f, 0.5f,
          2.5f, 3.5f, 1.5f, 0.0f, 1.0f, 0.0f,
      },
      /*labels=*/{3.5f, 3.5f, 6.5f, 2.0f, 9.0f, 4.0f, 7.0f, 1.0f},
      /*nrow=*/8,
      /*ncol=*/3,
  };
}

xgb_dmatrix_t make_train_matrix(const RegressionFixture& f) {
  xgb_dmatrix_t dtrain =
      xgb_dmatrix_create_from_mat(f.features.data(), f.nrow, f.ncol, -1.0f);
  if (dtrain) {
    xgb_dmatrix_set_float_info(dtrain, "label", f.labels.data(),
                               f.labels.size());
  }
  return dtrain;
}

}  // namespace

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
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, preds.size(), preds.data(),
                                &out_len),
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
  EXPECT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, 0, nullptr, &out_len), 2);
  EXPECT_EQ(out_len, f.nrow);

  // Capacity smaller than required: still rc=2, no copy.
  std::vector<float> small(f.nrow - 1, -42.0f);
  out_len = 0;
  EXPECT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, small.size(), small.data(),
                                &out_len),
            2);
  EXPECT_EQ(out_len, f.nrow);
  for (float v : small) EXPECT_FLOAT_EQ(v, -42.0f);

  // Resize and retry — should succeed now.
  std::vector<float> full(out_len, 0.0f);
  EXPECT_EQ(
      xgb_booster_predict(b, dtrain, kPredictConfig, full.size(), full.data(), &out_len),
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
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, first.size(), first.data(),
                                &len),
            0);
  ASSERT_EQ(xgb_booster_predict(b, dtrain, kPredictConfig, second.size(), second.data(),
                                &len),
            0);
  for (std::size_t i = 0; i < f.nrow; ++i) {
    EXPECT_FLOAT_EQ(first[i], second[i]) << "i=" << i;
  }

  xgb_booster_free(b);
  xgb_dmatrix_free(dtrain);
}

// ---------------------------------------------------------------------------
// Booster save/load tests
// ---------------------------------------------------------------------------

namespace {

// Train a small regressor for serialization tests.  Returns owned handles;
// callers must free both.
struct TrainedModel {
  xgb_booster_t booster;
  xgb_dmatrix_t dtrain;
  std::vector<float> baseline_predictions;
};

TrainedModel train_small_regressor(int rounds) {
  TrainedModel out{};
  const auto f = make_regression_fixture();
  out.dtrain = make_train_matrix(f);
  if (!out.dtrain) return out;
  const xgb_dmatrix_t cache[] = {out.dtrain};
  out.booster = xgb_booster_create(cache, 1);
  if (!out.booster) return out;
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

std::string make_temp_path(const char* suffix) {
  // testing::TempDir is gtest's per-run scratch space; cleaned up by the
  // harness.  Suffix controls the format XGBoost picks (json/ubj).
  return ::testing::TempDir() + "xgbcompat_model_" +
         std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
         suffix;
}

}  // namespace

TEST(XgbBoosterSerdeTest, SaveLoadFileRoundTrip) {
  TrainedModel m = train_small_regressor(20);
  ASSERT_NE(m.booster, nullptr);
  ASSERT_NE(m.dtrain, nullptr);

  const std::string path = make_temp_path(".json");
  ASSERT_EQ(xgb_booster_save_model(m.booster, path.c_str()), 0)
      << "last error: " << xgb_last_error();

  // Fresh booster, load, predict, compare.
  xgb_booster_t loaded = xgb_booster_create(nullptr, 0);
  ASSERT_NE(loaded, nullptr);
  ASSERT_EQ(xgb_booster_load_model(loaded, path.c_str()), 0)
      << "last error: " << xgb_last_error();

  std::vector<float> preds(m.baseline_predictions.size(), 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig,
                                preds.size(), preds.data(), &len),
            0);
  for (std::size_t i = 0; i < preds.size(); ++i) {
    EXPECT_FLOAT_EQ(preds[i], m.baseline_predictions[i]) << "i=" << i;
  }

  xgb_booster_free(loaded);
  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterSerdeTest, SaveToBufferTooSmallReportsRequiredSize) {
  TrainedModel m = train_small_regressor(5);
  ASSERT_NE(m.booster, nullptr);

  // Size probe: capacity 0, NULL buffer.
  std::uint64_t need = 0;
  EXPECT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             0, nullptr, &need),
            2);
  EXPECT_GT(need, 0u);

  // Capacity smaller than required: still rc=2, no copy.
  std::vector<char> small(need - 1, '\0');
  std::uint64_t need2 = 0;
  EXPECT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             small.size(), small.data(),
                                             &need2),
            2);
  EXPECT_EQ(need2, need);

  // Resize and succeed.
  std::vector<char> full(need, '\0');
  EXPECT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             full.size(), full.data(),
                                             &need2),
            0);
  EXPECT_EQ(need2, need);

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterSerdeTest, SaveLoadBufferRoundTrip) {
  TrainedModel m = train_small_regressor(20);
  ASSERT_NE(m.booster, nullptr);

  // Size, then serialize.
  std::uint64_t need = 0;
  ASSERT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             0, nullptr, &need),
            2);
  std::vector<char> buf(need, '\0');
  std::uint64_t got = 0;
  ASSERT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             buf.size(), buf.data(), &got),
            0);
  ASSERT_EQ(got, need);

  // Deserialize into a fresh booster and compare predictions.
  xgb_booster_t loaded = xgb_booster_create(nullptr, 0);
  ASSERT_NE(loaded, nullptr);
  ASSERT_EQ(xgb_booster_load_model_from_buffer(loaded, buf.data(), buf.size()),
            0)
      << "last error: " << xgb_last_error();

  std::vector<float> preds(m.baseline_predictions.size(), 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig,
                                preds.size(), preds.data(), &len),
            0);
  for (std::size_t i = 0; i < preds.size(); ++i) {
    EXPECT_FLOAT_EQ(preds[i], m.baseline_predictions[i]) << "i=" << i;
  }

  xgb_booster_free(loaded);
  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

// ---------------------------------------------------------------------------
// Booster eval-one-iter
// ---------------------------------------------------------------------------

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
  EXPECT_EQ(xgb_booster_eval_one_iter(m.booster, 0, dmats, names, 1,
                                      0, nullptr, &need),
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
  ASSERT_EQ(xgb_booster_eval_one_iter(b, 2, dmats, names, 2,
                                      buf.size(), buf.data(), &len),
            0);
  const std::string line(buf.data(), len);
  EXPECT_NE(line.find("train-"), std::string::npos) << line;
  EXPECT_NE(line.find("eval-"), std::string::npos) << line;

  xgb_booster_free(b);
  xgb_dmatrix_free(deval);
  xgb_dmatrix_free(dtrain);
}

TEST(XgbBoosterSerdeTest, LoadFromGarbageBufferFails) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr);
  const std::vector<char> garbage(64, 'x');
  EXPECT_NE(
      xgb_booster_load_model_from_buffer(b, garbage.data(), garbage.size()),
      0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
  xgb_booster_free(b);
}

TEST(XgbDMatrixTest, LeakSmokeTest) {
  // Allocate + free 10k DMatrices; RSS must not balloon.  ru_maxrss units
  // differ by platform (macOS = bytes, Linux = KB).
  struct rusage before{};
  struct rusage after{};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &before), 0);

  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> labels = {0.0f, 1.0f};
  constexpr int iters = 10000;
  for (int i = 0; i < iters; ++i) {
    xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 3, -1.0f);
    ASSERT_NE(h, nullptr);
    ASSERT_EQ(xgb_dmatrix_set_float_info(h, "label", labels.data(),
                                         labels.size()),
              0);
    xgb_dmatrix_free(h);
  }

  ASSERT_EQ(getrusage(RUSAGE_SELF, &after), 0);
#ifdef __APPLE__
  // ru_maxrss is in bytes on macOS.
  const std::int64_t bytes_growth =
      static_cast<std::int64_t>(after.ru_maxrss) -
      static_cast<std::int64_t>(before.ru_maxrss);
#else
  // ru_maxrss is in kilobytes on Linux.
  const std::int64_t bytes_growth =
      (static_cast<std::int64_t>(after.ru_maxrss) -
       static_cast<std::int64_t>(before.ru_maxrss)) *
      1024;
#endif
  // Smoke threshold: well below the ~600MB a leaking implementation would show.
  constexpr std::int64_t kMaxGrowthBytes = 50 * 1024 * 1024;
  EXPECT_LT(bytes_growth, kMaxGrowthBytes)
      << "RSS grew by " << bytes_growth << " bytes over " << iters
      << " create/free cycles";
}

}  // namespace
