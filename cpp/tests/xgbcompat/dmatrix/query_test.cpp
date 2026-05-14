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

TEST(XgbDMatrixTest, NumRowNumCol) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
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

TEST(XgbDMatrixTest, NumNonMissingAndCsrRoundTrip) {
  const std::vector<float> data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
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
  ASSERT_EQ(xgb_dmatrix_get_data_as_csr(h, "{}", indptr.data(), indices.data(),
                                        out_data.data()),
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

}  // namespace
