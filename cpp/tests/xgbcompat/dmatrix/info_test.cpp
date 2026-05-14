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

TEST(XgbDMatrixTest, SetFloatInfoSuccess) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> labels = {0.0f, 1.0f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  const int rc =
      xgb_dmatrix_set_float_info(h, "label", labels.data(), labels.size());
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

TEST(XgbDMatrixTest, GetFloatInfoRoundTrip) {
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> labels = {0.5f, 1.5f};
  xgb_dmatrix_t h = xgb_dmatrix_create_from_mat(data.data(), 2, 2, -1.0f);
  ASSERT_NE(h, nullptr);
  ASSERT_EQ(
      xgb_dmatrix_set_float_info(h, "label", labels.data(), labels.size()), 0);
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

  ASSERT_EQ(xgb_dmatrix_set_uint_info(h, "group", group.data(), group.size()),
            0)
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
  ASSERT_EQ(xgb_dmatrix_set_info_from_interface(h, "label", label_ai.c_str()),
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

}  // namespace
