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
