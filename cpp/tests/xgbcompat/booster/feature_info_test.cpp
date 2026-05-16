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

TEST(XgbBoosterTest, FeatureInfoDumpAndScores) {
  TrainedModel m = train_small_regressor(5);
  ASSERT_NE(m.booster, nullptr);

  const char* names[] = {"f0", "f1", "f2"};
  const char* types[] = {"q", "q", "q"};
  ASSERT_EQ(
      xgb_booster_set_str_feature_info(m.booster, "feature_name", names, 3), 0)
      << "last error: " << xgb_last_error();
  ASSERT_EQ(
      xgb_booster_set_str_feature_info(m.booster, "feature_type", types, 3), 0)
      << "last error: " << xgb_last_error();

  std::uint64_t feature_info_len = 0;
  std::uint64_t feature_info_count = 0;
  ASSERT_EQ(
      xgb_booster_get_str_feature_info(m.booster, "feature_name", 0, nullptr,
                                       &feature_info_len, &feature_info_count),
      2);
  EXPECT_EQ(feature_info_count, 3u);

  std::uint64_t dump_len = 0;
  std::uint64_t dump_count = 0;
  ASSERT_EQ(xgb_booster_dump_model(m.booster, "json", 0, 0, nullptr, &dump_len,
                                   &dump_count),
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
  ASSERT_EQ(xgb_booster_dump_model_with_features(
                m.booster, names, types, 3, "text", 0, named_dump.size(),
                named_dump.data(), &dump_len, &dump_count),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_NE(std::string(named_dump.data(), dump_len).find("f"),
            std::string::npos);

  std::uint64_t feature_bytes = 0;
  std::uint64_t n_features = 0;
  std::uint64_t dim = 0;
  std::uint64_t n_scores = 0;
  const char* config =
      "{\"importance_type\":\"weight\",\"feature_names\":[\"f0\",\"f1\",\"f2\"]"
      "}";
  ASSERT_EQ(xgb_booster_feature_score(m.booster, config, 0, nullptr,
                                      &feature_bytes, &n_features, 0, nullptr,
                                      &dim, 0, nullptr, &n_scores),
            2)
      << "last error: " << xgb_last_error();
  ASSERT_GT(n_features, 0u);
  ASSERT_GT(dim, 0u);
  ASSERT_GT(n_scores, 0u);
  std::vector<char> score_features(feature_bytes, '\0');
  std::vector<std::uint64_t> shape(dim, 0);
  std::vector<float> scores(n_scores, 0.0f);
  ASSERT_EQ(xgb_booster_feature_score(
                m.booster, config, score_features.size(), score_features.data(),
                &feature_bytes, &n_features, shape.size(), shape.data(), &dim,
                scores.size(), scores.data(), &n_scores),
            0)
      << "last error: " << xgb_last_error();
  EXPECT_GT(scores[0], 0.0f);

  xgb_booster_free(m.booster);
  xgb_dmatrix_free(m.dtrain);
}

}  // namespace
