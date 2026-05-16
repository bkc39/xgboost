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

TEST(XgbDMatrixTest, CreateFreeRoundTrip) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
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
  xgb_dmatrix_t h =
      xgb_dmatrix_create_from_csr(indptr_ai.c_str(), indices_ai.c_str(),
                                  data_ai.c_str(), 3, "{\"missing\":-1.0}");
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
  xgb_dmatrix_t h =
      xgb_dmatrix_create_from_csc(indptr_ai.c_str(), indices_ai.c_str(),
                                  data_ai.c_str(), 2, "{\"missing\":-1.0}");
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

TEST(XgbDMatrixTest, SliceRows) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
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

}  // namespace
