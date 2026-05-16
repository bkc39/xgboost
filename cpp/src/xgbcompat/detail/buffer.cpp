#include "xgbcompat/detail/buffer.hpp"

#include <algorithm>
#include <cstring>

namespace xgbcompat::detail {

uint64_t nul_separated_size(const char** values, uint64_t count) {
  uint64_t need = 0;
  for (uint64_t i = 0; i < count; ++i) {
    need += static_cast<uint64_t>(std::strlen(values[i])) + 1;
  }
  return need;
}

void copy_nul_separated(const char** values, uint64_t count, char* out_buffer) {
  char* cursor = out_buffer;
  for (uint64_t i = 0; i < count; ++i) {
    const size_t len_i = std::strlen(values[i]);
    std::copy(values[i], values[i] + len_i, cursor);
    cursor += len_i;
    *cursor++ = '\0';
  }
}

int copy_float_result(const float* values, uint64_t count,
                      uint64_t buffer_capacity, float* out_buffer,
                      uint64_t* out_len) {
  *out_len = count;
  if (buffer_capacity < count) {
    return 2;
  }
  if (count > 0 && values && out_buffer) {
    std::copy(values, values + count, out_buffer);
  }
  return 0;
}

}  // namespace xgbcompat::detail
