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

TEST(XgbCompatTest, VersionLooksLikeSemver) {
  const std::string v = xgbcompat::version();
  EXPECT_FALSE(v.empty());
  const std::regex semver(R"(^\d+\.\d+\.\d+$)");
  EXPECT_TRUE(std::regex_match(v, semver)) << "got: " << v;
}

TEST(XgbCompatTest, ExternCVersionMatchesCpp) {
  const std::string cpp_v = xgbcompat::version();
  const char* c_v = xgb_version();
  ASSERT_NE(c_v, nullptr);
  EXPECT_EQ(cpp_v, std::string(c_v));
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

}  // namespace
