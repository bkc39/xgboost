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
  ASSERT_EQ(xgb_booster_get_attr(b, "owner", value.size(), value.data(), &len,
                                 &found),
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

}  // namespace
