#include "xgbcompat/c_api.h"

int xgbcompat_c_api_header_smoke(void) {
  return xgb_version() == 0 ? 0 : 0;
}
