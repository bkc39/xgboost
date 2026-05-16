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
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig, preds.size(),
                                preds.data(), &len),
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
  EXPECT_EQ(
      xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                       small.size(), small.data(), &need2),
      2);
  EXPECT_EQ(need2, need);

  // Resize and succeed.
  std::vector<char> full(need, '\0');
  EXPECT_EQ(xgb_booster_save_model_to_buffer(m.booster, "{\"format\":\"ubj\"}",
                                             full.size(), full.data(), &need2),
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
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig, preds.size(),
                                preds.data(), &len),
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
  ASSERT_EQ(
      xgb_booster_serialize_to_buffer(m.booster, buf.size(), buf.data(), &got),
      0);
  ASSERT_EQ(got, need);

  // Restore into a fresh booster and verify predictions match.
  xgb_booster_t loaded = xgb_booster_create(nullptr, 0);
  ASSERT_NE(loaded, nullptr);
  ASSERT_EQ(xgb_booster_unserialize_from_buffer(loaded, buf.data(), buf.size()),
            0)
      << "last error: " << xgb_last_error();

  std::vector<float> preds(m.baseline_predictions.size(), 0.0f);
  std::uint64_t len = 0;
  ASSERT_EQ(xgb_booster_predict(loaded, m.dtrain, kPredictConfig, preds.size(),
                                preds.data(), &len),
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
  ASSERT_EQ(
      xgb_booster_predict(loaded, m.dtrain, kPredictConfig,
                          restored_after.size(), restored_after.data(), &len),
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
  EXPECT_NE(
      xgb_booster_unserialize_from_buffer(b, garbage.data(), garbage.size()),
      0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
  xgb_booster_free(b);
}

TEST(XgbBoosterSerdeTest, LoadFromGarbageBufferFails) {
  xgb_booster_t b = xgb_booster_create(nullptr, 0);
  ASSERT_NE(b, nullptr);
  const std::vector<char> garbage(64, 'x');
  EXPECT_NE(
      xgb_booster_load_model_from_buffer(b, garbage.data(), garbage.size()), 0);
  EXPECT_FALSE(std::string(xgb_last_error()).empty());
  xgb_booster_free(b);
}

}  // namespace
