#pragma once

#include <stdint.h>

namespace xgbcompat::detail {

uint64_t nul_separated_size(const char** values, uint64_t count);
void copy_nul_separated(const char** values, uint64_t count, char* out_buffer);
int copy_float_result(const float* values, uint64_t count,
                      uint64_t buffer_capacity, float* out_buffer,
                      uint64_t* out_len);

}  // namespace xgbcompat::detail

namespace xgbcompat {
using detail::copy_float_result;
using detail::copy_nul_separated;
using detail::nul_separated_size;
}  // namespace xgbcompat
