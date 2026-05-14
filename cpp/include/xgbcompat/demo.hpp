#pragma once
#include <string>
#include <vector>

namespace xgbcompat {

struct DemoResult {
  std::vector<float> predictions;
};

std::string version();

DemoResult run_regression_demo();

DemoResult run_classification_demo();

std::string last_error();

}  // namespace xgbcompat
