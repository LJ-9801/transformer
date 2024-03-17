#ifndef UTILITIES_H
#define UTILITIES_H
#include <cstdint>
#include <cstddef>

static inline int index_from_stride(const uint32_t* indices, const uint32_t* strides, uint32_t dim) {
  int index = 0;
  for (size_t i = 0; i < dim; ++i) {
    index += indices[i] * strides[i];
  }
  return index;
}

static inline int index_from_shape(const uint32_t* indices, const uint32_t* shape, uint32_t dim){
  int index = 0;
  int stride = 1;
  for (int i = dim - 1; i >= 0; --i) {
      index += indices[i] * stride;
      stride *= shape[i];
  }
  return index;
}
#endif