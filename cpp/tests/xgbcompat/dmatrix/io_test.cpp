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
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
  };
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 3, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const std::vector<float> labels = {1.0f, 3.0f, 5.0f};
  ASSERT_EQ(
      xgb_dmatrix_set_float_info(h, "label", labels.data(), labels.size()), 0);

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
  ASSERT_EQ(xgb_dmatrix_get_quantile_cut(h, "{}", 0, nullptr, &indptr_len, 0,
                                         nullptr, &data_len),
            2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(indptr_len, 0u);
  ASSERT_GT(data_len, 0u);
  std::vector<char> indptr(indptr_len, '\0');
  std::vector<char> cuts(data_len, '\0');
  ASSERT_EQ(xgb_dmatrix_get_quantile_cut(h, "{}", indptr.size(), indptr.data(),
                                         &indptr_len, cuts.size(), cuts.data(),
                                         &data_len),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(std::string(indptr.data(), indptr_len).find("\"shape\""),
            std::string::npos);
  EXPECT_NE(std::string(cuts.data(), data_len).find("\"shape\""),
            std::string::npos);

  xgb_booster_free(b);
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
    ASSERT_EQ(
        xgb_dmatrix_set_float_info(h, "label", labels.data(), labels.size()),
        0);
    xgb_dmatrix_free(h);
  }

  ASSERT_EQ(getrusage(RUSAGE_SELF, &after), 0);
#ifdef __APPLE__
  // ru_maxrss is in bytes on macOS.
  const std::int64_t bytes_growth = static_cast<std::int64_t>(after.ru_maxrss) -
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
