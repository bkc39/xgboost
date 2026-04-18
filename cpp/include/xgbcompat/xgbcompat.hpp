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

#ifdef __cplusplus
extern "C" {
#endif

const char* xgb_version(void);

const char* xgb_last_error(void);

int xgb_run_regression_demo(double* out_first_prediction);

int xgb_run_classification_demo(double* out_first_prediction);

#ifdef __cplusplus
}
#endif
