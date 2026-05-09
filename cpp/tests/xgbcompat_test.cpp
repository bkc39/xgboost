#include "xgbcompat/xgbcompat.hpp"

#include <gtest/gtest.h>

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
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

std::string array_interface(std::uintptr_t ptr,
                            const std::string& typestr,
                            std::initializer_list<std::uint64_t> shape) {
  std::ostringstream os;
  os << "{\"data\":[" << ptr << ",false],\"typestr\":\"" << typestr
     << "\",\"shape\":[";
  bool first = true;
  for (std::uint64_t dim : shape) {
    if (!first) os << ",";
    first = false;
    os << dim;
  }
  os << "],\"version\":3}";
  return os.str();
}

TEST(XgbDMatrixTest, CreateFromDenseArrayInterface) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::string ai = array_interface(
      reinterpret_cast<std::uintptr_t>(data.data()), "<f4", {2, 3});
  xgb_dmatrix_t h =
      xgb_dmatrix_create_from_dense(ai.c_str(), "{\"missing\":-1.0}");
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 3u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, CreateFromCSRArrayInterfaces) {
  std::vector<std::uint64_t> indptr = {0, 2, 4};
  std::vector<std::uint32_t> indices = {0, 2, 1, 2};
  std::vector<float> data = {1.0f, 3.0f, 5.0f, 6.0f};
  const std::string indptr_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indptr.data()), "<u8", {3});
  const std::string indices_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indices.data()), "<u4", {4});
  const std::string data_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(data.data()), "<f4", {4});
  xgb_dmatrix_t h = xgb_dmatrix_create_from_csr(
      indptr_ai.c_str(), indices_ai.c_str(), data_ai.c_str(), 3,
      "{\"missing\":-1.0}");
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 3u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, CreateFromCSCArrayInterfaces) {
  std::vector<std::uint64_t> indptr = {0, 1, 2, 4};
  std::vector<std::uint32_t> indices = {0, 1, 0, 1};
  std::vector<float> data = {1.0f, 5.0f, 3.0f, 6.0f};
  const std::string indptr_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indptr.data()), "<u8", {4});
  const std::string indices_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indices.data()), "<u4", {4});
  const std::string data_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(data.data()), "<f4", {4});
  xgb_dmatrix_t h = xgb_dmatrix_create_from_csc(
      indptr_ai.c_str(), indices_ai.c_str(), data_ai.c_str(), 2,
      "{\"missing\":-1.0}");
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 3u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, CreateFromColumnarArrayInterfaces) {
  std::vector<float> col0 = {1.0f, 4.0f};
  std::vector<float> col1 = {2.0f, 5.0f};
  std::vector<float> col2 = {3.0f, 6.0f};
  const std::string col0_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col0.data()), "<f4", {2});
  const std::string col1_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col1.data()), "<f4", {2});
  const std::string col2_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(col2.data()), "<f4", {2});
  const std::string data = "[" + col0_ai + "," + col1_ai + "," + col2_ai + "]";
  xgb_dmatrix_t h =
      xgb_dmatrix_create_from_columnar(data.c_str(), "{\"missing\":-1.0}");
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 3u);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, CreateFromUriLibsvm) {
  const std::string path = "xgbcompat-libsvm-test.txt";
  {
    std::ofstream out(path);
    out << "0 1:1 3:3\n";
    out << "1 2:5 3:6\n";
  }
  const std::string config =
      "{\"uri\":\"" + path + "?format=libsvm\",\"silent\":1}";
  xgb_dmatrix_t h = xgb_dmatrix_create_from_uri(config.c_str());
  ASSERT_NE(h, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(h, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(h, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 4u);
  xgb_dmatrix_free(h);
  std::remove(path.c_str());
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

TEST(XgbDMatrixTest, StringFeatureInfoRoundTrip) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);

  const char* names[] = {"height", "weight"};
  const char* types[] = {"q", "q"};
  ASSERT_EQ(xgb_dmatrix_set_str_feature_info(h, "feature_name", names, 2), 0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(xgb_dmatrix_set_str_feature_info(h, "feature_type", types, 2), 0)
      << "last error: " << xgb_last_error();

  std::uint64_t len = 0;
  std::uint64_t count = 0;
  ASSERT_EQ(xgb_dmatrix_get_str_feature_info(h, "feature_name", 0, nullptr,
                                             &len, &count),
            2);
  EXPECT_EQ(count, 2u);
  std::vector<char> buf(len, '\0');
  ASSERT_EQ(xgb_dmatrix_get_str_feature_info(h, "feature_name", buf.size(),
                                             buf.data(), &len, &count),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(count, 2u);
  const std::string first(buf.data());
  const std::string second(buf.data() + first.size() + 1);
  EXPECT_EQ(first, "height");
  EXPECT_EQ(second, "weight");

  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, UIntInfoRoundTrip) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<std::uint32_t> group = {2u};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);

  ASSERT_EQ(xgb_dmatrix_set_uint_info(h, "group", group.data(), group.size()), 0)
      << "last error: " << xgb_last_error();
  std::uint64_t len = 0;
  const std::uint32_t* out = nullptr;
  ASSERT_EQ(xgb_dmatrix_get_uint_info(h, "group_ptr", &len, &out), 0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(len, 2u);
  ASSERT_NE(out, nullptr);
  EXPECT_EQ(out[0], 0u);
  EXPECT_EQ(out[1], 2u);

  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, SetInfoFromInterfaceRoundTripsLabels) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> labels = {0.25f, 0.75f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const std::string label_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(labels.data()), "<f4", {2});
  ASSERT_EQ(xgb_dmatrix_set_info_from_interface(h, "label",
                                                label_ai.c_str()),
            0)
      << "last error: " << xgb_last_error();

  std::uint64_t len = 0;
  const float* out = nullptr;
  ASSERT_EQ(xgb_dmatrix_get_float_info(h, "label", &len, &out), 0);
  ASSERT_EQ(len, 2u);
  EXPECT_FLOAT_EQ(out[0], 0.25f);
  EXPECT_FLOAT_EQ(out[1], 0.75f);

  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, SliceRows) {
  const std::vector<float> data = {
      1.0f, 2.0f,
      3.0f, 4.0f,
      5.0f, 6.0f,
  };
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 3, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const std::vector<std::int32_t> idx = {2, 0};
  xgb_dmatrix_t sliced = xgb_dmatrix_slice(h, idx.data(), idx.size(), 0);
  ASSERT_NE(sliced, nullptr) << "last error: " << xgb_last_error();

  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  ASSERT_EQ(xgb_dmatrix_num_row(sliced, &nrow), 0);
  ASSERT_EQ(xgb_dmatrix_num_col(sliced, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 2u);

  std::uint64_t nnz = 0;
  ASSERT_EQ(xgb_dmatrix_num_non_missing(sliced, &nnz), 0);
  std::vector<std::uint64_t> indptr(nrow + 1, 0);
  std::vector<std::uint32_t> indices(nnz, 0);
  std::vector<float> out_data(nnz, 0.0f);
  ASSERT_EQ(xgb_dmatrix_get_data_as_csr(sliced, "{}", indptr.data(),
                                        indices.data(), out_data.data()),
            0);
  ASSERT_EQ(out_data.size(), 4u);
  EXPECT_FLOAT_EQ(out_data[0], 5.0f);
  EXPECT_FLOAT_EQ(out_data[1], 6.0f);
  EXPECT_FLOAT_EQ(out_data[2], 1.0f);
  EXPECT_FLOAT_EQ(out_data[3], 2.0f);

  xgb_dmatrix_free(sliced);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, SaveBinaryAndReload) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const std::string path = ::testing::TempDir() + "xgbcompat_dmatrix.buffer";
  ASSERT_EQ(xgb_dmatrix_save_binary(h, path.c_str(), 1), 0)
      << "last error: " << xgb_last_error();

  const std::string config = "{\"uri\":\"" + path + "\",\"silent\":1}";
  xgb_dmatrix_t loaded = xgb_dmatrix_create_from_uri(config.c_str());
  ASSERT_NE(loaded, nullptr) << "last error: " << xgb_last_error();
  std::uint64_t nrow = 0;
  std::uint64_t ncol = 0;
  EXPECT_EQ(xgb_dmatrix_num_row(loaded, &nrow), 0);
  EXPECT_EQ(xgb_dmatrix_num_col(loaded, &ncol), 0);
  EXPECT_EQ(nrow, 2u);
  EXPECT_EQ(ncol, 2u);

  xgb_dmatrix_free(loaded);
  xgb_dmatrix_free(h);
}

TEST(XgbDMatrixTest, QuantileCutReturnsArrayInterfaces) {
  const std::vector<float> data = {
      1.0f, 2.0f,
      3.0f, 4.0f,
      5.0f, 6.0f,
  };
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 3, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const std::vector<float> labels = {1.0f, 3.0f, 5.0f};
  ASSERT_EQ(xgb_dmatrix_set_float_info(h, "label", labels.data(),
                                       labels.size()),
            0);

  const xgb_dmatrix_t cache[] = {h};
  xgb_booster_t b = xgb_booster_create(cache, 1);
  ASSERT_NE(b, nullptr) << "last error: " << xgb_last_error();
  ASSERT_EQ(xgb_booster_set_param(b, "objective", "reg:squarederror"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "tree_method", "hist"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "max_depth", "2"), 0);
  ASSERT_EQ(xgb_booster_set_param(b, "verbosity", "0"), 0);
  ASSERT_EQ(xgb_booster_update_one_iter(b, 0, h), 0)
      << "last error: " << xgb_last_error();

  std::uint64_t indptr_len = 0;
  std::uint64_t data_len = 0;
  ASSERT_EQ(xgb_dmatrix_get_quantile_cut(h, "{}", 0, nullptr, &indptr_len,
                                         0, nullptr, &data_len),
            2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(indptr_len, 0u);
  ASSERT_GT(data_len, 0u);
  std::vector<char> indptr(indptr_len, '\0');
  std::vector<char> cuts(data_len, '\0');
  ASSERT_EQ(xgb_dmatrix_get_quantile_cut(h, "{}", indptr.size(),
                                         indptr.data(), &indptr_len,
                                         cuts.size(), cuts.data(), &data_len),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(std::string(indptr.data(), indptr_len).find("\"shape\""),
            std::string::npos);
  EXPECT_NE(std::string(cuts.data(), data_len).find("\"shape\""),
            std::string::npos);

  xgb_booster_free(b);
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

constexpr const char* kInplacePredictConfig =
    "{\"type\":0,\"training\":false,"
    "\"iteration_begin\":0,\"iteration_end\":0,"
    "\"strict_shape\":false,\"missing\":-1.0}";

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

std::vector<float> predict_with_dmatrix(xgb_booster_t booster,
                                        xgb_dmatrix_t dmat,
                                        std::size_t nrow) {
  std::vector<float> preds(nrow, 0.0f);
  std::uint64_t len = 0;
  xgb_booster_predict(booster, dmat, kPredictConfig, preds.size(),
                      preds.data(), &len);
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

TEST(XgbBoosterTest, PredictFromDenseMatchesDMatrix) {
  TrainedModel m = train_small_regressor(12);
  ASSERT_NE(m.booster, nullptr);
  ASSERT_NE(m.dtrain, nullptr);
  const auto f = make_regression_fixture();
  const std::string data_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(f.features.data()), "<f4",
      {f.nrow, f.ncol});

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict_from_dense(
                m.booster, data_ai.c_str(), kInplacePredictConfig, nullptr,
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
  const std::string indices_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(indices.data()), "<u4", {indices.size()});
  const std::string data_ai = array_interface(
      reinterpret_cast<std::uintptr_t>(f.features.data()), "<f4",
      {f.features.size()});

  std::vector<float> preds(f.nrow, 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict_from_csr(
                m.booster, indptr_ai.c_str(), indices_ai.c_str(),
                data_ai.c_str(), f.ncol, kInplacePredictConfig, nullptr,
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
  ASSERT_EQ(xgb_booster_predict_from_columnar(
                m.booster, data.c_str(), kInplacePredictConfig, nullptr,
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

TEST(XgbBoosterTest, JsonConfigRoundTrip) {
  TrainedModel m = train_small_regressor(2);
  ASSERT_NE(m.booster, nullptr);

  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_save_json_config(m.booster, 0, nullptr, &len), 2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(len, 0u);
  std::vector<char> config(len, '\0');
  ASSERT_EQ(xgb_booster_save_json_config(m.booster, config.size(),
                                         config.data(), &len),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_EQ(config.front(), '{');

  xgb_booster_t loaded = xgb_booster_create(nullptr, 0);
  ASSERT_NE(loaded, nullptr);
  ASSERT_EQ(xgb_booster_load_json_config(
                loaded, std::string(config.data(), len).c_str()),
            0)
      << "last error: " << xgb_last_error();

  xgb_booster_free(loaded);
  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterTest, AttrsRoundTripAndDelete) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(xgb_booster_set_attr(b, "owner", "racket"), 0)
      << "last error: " << xgb_last_error();

  std::uint64_t len = 0;
  int found = 0;
  ASSERT_EQ(xgb_booster_get_attr(b, "owner", 0, nullptr, &len, &found), 2)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(found, 1);
  std::vector<char> value(len, '\0');
  ASSERT_EQ(xgb_booster_get_attr(b, "owner", value.size(), value.data(),
                                 &len, &found),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_EQ(std::string(value.data(), len), "racket");

  std::uint64_t names_len = 0;
  std::uint64_t count = 0;
  ASSERT_EQ(xgb_booster_get_attr_names(b, 0, nullptr, &names_len, &count), 2)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(count, 1u);
  std::vector<char> names(names_len, '\0');
  ASSERT_EQ(xgb_booster_get_attr_names(b, names.size(), names.data(),
                                       &names_len, &count),
            0);
  EXPECT_EQ(std::string(names.data()), "owner");

  ASSERT_EQ(xgb_booster_delete_attr(b, "owner"), 0)
      << "last error: " << xgb_last_error();
  len = 0;
  found = 1;
  ASSERT_EQ(xgb_booster_get_attr(b, "owner", 0, nullptr, &len, &found), 0);
  EXPECT_EQ(found, 0);

  xgb_booster_free(b);
}

TEST(XgbBoosterTest, FeatureInfoDumpAndScores) {
  TrainedModel m = train_small_regressor(5);
  ASSERT_NE(m.booster, nullptr);

  const char* names[] = {"f0", "f1", "f2"};
  const char* types[] = {"q", "q", "q"};
  ASSERT_EQ(xgb_booster_set_str_feature_info(m.booster, "feature_name",
                                             names, 3),
            0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(xgb_booster_set_str_feature_info(m.booster, "feature_type",
                                             types, 3),
            0)
      << "last error: " << xgb_last_error();

  std::uint64_t feature_info_len = 0;
  std::uint64_t feature_info_count = 0;
  ASSERT_EQ(xgb_booster_get_str_feature_info(m.booster, "feature_name",
                                             0, nullptr, &feature_info_len,
                                             &feature_info_count),
            2);
  EXPECT_EQ(feature_info_count, 3u);

  std::uint64_t dump_len = 0;
  std::uint64_t dump_count = 0;
  ASSERT_EQ(xgb_booster_dump_model(m.booster, "json", 0, 0, nullptr,
                                   &dump_len, &dump_count),
            2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(dump_len, 0u);
  ASSERT_GT(dump_count, 0u);
  std::vector<char> dump(dump_len, '\0');
  ASSERT_EQ(xgb_booster_dump_model(m.booster, "json", 0, dump.size(),
                                   dump.data(), &dump_len, &dump_count),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(std::string(dump.data()).find("{"), std::string::npos);

  dump_len = 0;
  dump_count = 0;
  ASSERT_EQ(xgb_booster_dump_model_with_features(m.booster, names, types, 3,
                                                 "text", 0, 0, nullptr,
                                                 &dump_len, &dump_count),
            2)
      << "last error: " << xgb_last_error();
  std::vector<char> named_dump(dump_len, '\0');
  ASSERT_EQ(xgb_booster_dump_model_with_features(m.booster, names, types, 3,
                                                 "text", 0,
                                                 named_dump.size(),
                                                 named_dump.data(),
                                                 &dump_len, &dump_count),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(std::string(named_dump.data(), dump_len).find("f"),
            std::string::npos);

  std::uint64_t feature_bytes = 0;
  std::uint64_t n_features = 0;
  std::uint64_t dim = 0;
  std::uint64_t n_scores = 0;
  const char* config =
      "{\"importance_type\":\"weight\",\"feature_names\":[\"f0\",\"f1\",\"f2\"]}";
  ASSERT_EQ(xgb_booster_feature_score(m.booster, config, 0, nullptr,
                                      &feature_bytes, &n_features,
                                      0, nullptr, &dim,
                                      0, nullptr, &n_scores),
            2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(n_features, 0u);
  ASSERT_GT(dim, 0u);
  ASSERT_GT(n_scores, 0u);
  std::vector<char> score_features(feature_bytes, '\0');
  std::vector<std::uint64_t> shape(dim, 0);
  std::vector<float> scores(n_scores, 0.0f);
  ASSERT_EQ(xgb_booster_feature_score(m.booster, config,
                                      score_features.size(),
                                      score_features.data(),
                                      &feature_bytes, &n_features,
                                      shape.size(), shape.data(), &dim,
                                      scores.size(), scores.data(),
                                      &n_scores),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_GT(scores[0], 0.0f);

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

// ---------------------------------------------------------------------------
// Booster save/load tests
// ---------------------------------------------------------------------------

namespace {

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

TEST(XgbBoosterSerdeTest, SerializeUnserializeRoundTrip) {
  // Snapshot includes training-state caches (not just the model), so a
  // restored booster can keep calling update_one_iter and produce the same
  // trajectory as the original.
  TrainedModel m = train_small_regressor(5);
  ASSERT_NE(m.booster, nullptr);

  // Size probe.
  std::uint64_t need = 0;
  ASSERT_EQ(xgb_booster_serialize_to_buffer(m.booster, 0, nullptr, &need), 2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(need, 0u);

  // Capacity smaller than required: still rc=2, no copy.
  std::vector<char> small(need - 1, '\0');
  std::uint64_t need2 = 0;
  ASSERT_EQ(xgb_booster_serialize_to_buffer(m.booster, small.size(),
                                            small.data(), &need2),
            2);
  EXPECT_EQ(need2, need);

  // Resize and serialize.
  std::vector<char> buf(need, '\0');
  std::uint64_t got = 0;
  ASSERT_EQ(xgb_booster_serialize_to_buffer(m.booster, buf.size(), buf.data(),
                                            &got),
            0);
  ASSERT_EQ(got, need);

  // Restore into a fresh booster and verify predictions match.
  xgb_booster_t loaded = xgb_booster_create(nullptr, 0);
  ASSERT_NE(loaded, nullptr);
  ASSERT_EQ(xgb_booster_unserialize_from_buffer(loaded, buf.data(),
                                                buf.size()),
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

  // Continue training on the restored booster and on the original booster
  // independently — the snapshot must put `loaded` in the same state, so
  // both should produce the same updates.
  ASSERT_EQ(xgb_booster_update_one_iter(m.booster, 5, m.dtrain), 0);
  ASSERT_EQ(xgb_booster_update_one_iter(loaded, 5, m.dtrain), 0);

  std::vector<float> orig_after(m.baseline_predictions.size(), 0.0f);
  std::vector<float> restored_after(m.baseline_predictions.size(), 0.0f);
  ASSERT_EQ(xgb_booster_predict(m.booster, m.dtrain, kPredictConfig,
                                orig_after.size(), orig_after.data(), &len),
            0);
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig,
                                restored_after.size(), restored_after.data(),
                                &len),
            0);
  for (std::size_t i = 0; i < orig_after.size(); ++i) {
    EXPECT_FLOAT_EQ(orig_after[i], restored_after[i]) << "i=" << i;
  }

  xgb_booster_free(loaded);
  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

TEST(XgbBoosterSerdeTest, UnserializeFromGarbageBufferFails) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr);
  const std::vector<char> garbage(64, 'x');
  EXPECT_NE(xgb_booster_unserialize_from_buffer(b, garbage.data(),
                                                garbage.size()),
            0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
  xgb_booster_free(b);
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
