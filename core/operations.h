#ifndef OPERATIONS_H
#define OPERATIONS_H
#include <cmath>
#include "common/kernels.h"
#include "tensor.h"


template <typename T>
Tensor<T> expand(const Tensor<T>& src, shape_t new_shape)
{

  Tensor<T> trg = Tensor<T>(new_shape);
  uint32_t src_size = src.size();
  uint32_t trg_size = trg.size();

  T* src_ptr = accessor<T>::const_ptr(src);
  T* trg_ptr = accessor<T>::get(trg);

  #pragma omp parallel for
  for (int i = 0; i < trg_size; ++i) {
    int src_index = 0;
    for(int j = 0; j < src.ndim(); j++){
      int idx = (i / trg.stride()[j]) % src.shape()[j];
      src_index += idx * src.stride()[j];
    }
    trg_ptr[i] = src_ptr[src_index];
  } 

  return trg;
}

// there's gotta be a better way to check batch_matmul compatibility
template <typename T>
Tensor<T> batch_matmul(const Tensor<T>& a, const Tensor<T>& b)
{
  if(a.empty() || b.empty()){
    return Tensor<T>();
  }

  if(b.ndim() > a.ndim()){
    assert(false && "The second tensor should have less or equal dimensions than the first tensor");
  }

  bool condition = false;
  //shape_t new_shape = shape_t();

  if(a.ndim() == b.ndim() && a.ndim() > 1){

    condition = a.shape()[a.ndim() - 1] != b.shape()[b.ndim() - 2];
    assert(!condition && "Matrix multiplication is not compatible");
    shape_t new_shape = shape_t();
    for(int i = 0; i < a.ndim() - 1; i++){
      new_shape.push_back(a.shape()[i]);
    }
    new_shape.push_back(b.shape()[b.ndim() - 1]);
    Tensor<T> out = Tensor<T>(new_shape);

    uint32_t batch_size = multiply_accumulate(new_shape.begin(), new_shape.end() - 2);
    uint32_t M = a.shape()[a.ndim() - 2];
    uint32_t N = b.shape()[b.ndim() - 1];
    uint32_t K = a.shape()[a.ndim() - 1];

    #pragma omp parallel for
    for(int i = 0; i < batch_size; i++){
      gemm<T>(accessor<T>::const_ptr(a) + i * M * K, 
              accessor<T>::const_ptr(b) + i * K * N, 
              accessor<T>::get(out) + i * M * N, 
              M, N, K);
    }

    return out;

  }else if(b.ndim() == 1 && a.ndim() > 1){
    // if b is 1 dimensional, then we are doing a dot product
    // check if the last dimension of two tensors are the same
    condition = a.shape()[a.ndim() - 1] != b.shape()[0];
    assert(!condition && "Matrix multiplication is not compatible");

    shape_t new_shape = shape_t();
    for(int i = 0; i < a.ndim() - 1; i++){
      new_shape.push_back(a.shape()[i]);
    }

    Tensor<T> out = Tensor<T>(new_shape);
    uint32_t batch_size = multiply_accumulate(new_shape.begin(), new_shape.end()); 
    uint32_t ddot_size = a.shape()[a.ndim() - 1];
    
    #pragma omp parallel for
    for(int i = 0; i < batch_size; i++){
      dot<T>(accessor<T>::const_ptr(a) + i * ddot_size, 
            accessor<T>::const_ptr(b), 
            accessor<T>::get(out) + i, 
            ddot_size);
    } 

    return out;
  }
  
  // in other case, we have A ndim > B ndim and B ndim > 1
  uint32_t M = a.shape()[a.ndim() - 2];
  uint32_t N = b.shape()[b.ndim() - 1];
  uint32_t K = a.shape()[a.ndim() - 1];

  shape_t new_shape = shape_t();
  for(int i = 0; i < a.ndim() - 1; i++){
    new_shape.push_back(a.shape()[i]);
  }
  new_shape.push_back(b.shape()[b.ndim() - 1]);
  Tensor<T> out = Tensor<T>(new_shape);
  uint32_t batch_size = multiply_accumulate(new_shape.begin(), new_shape.end() - 2);

  #pragma omp parallel for
  for(int i = 0; i < batch_size; i++){
    gemm<T>(accessor<T>::const_ptr(a) + i * M * K, 
            accessor<T>::const_ptr(b), 
            accessor<T>::get(out) + i * M * N, 
            M, N, K);
  }

  return out;
}

template <typename T>
Tensor<T> masked_fill(const Tensor<T>& src, const Tensor<bool>& mask, T value)
{
  if(src.empty() || mask.empty()){
    return Tensor<T>();
  }

  assert(src.shape() == mask.shape() && "The shape of the mask should be the same as the source tensor");

  Tensor<T> trg = Tensor<T>(src.shape()); 
  uint32_t src_size = src.size();

  #pragma omp parallel for
  for (int i = 0; i < src_size; ++i) {
    T* src_ptr = accessor<T>::const_ptr(src);
    T* trg_ptr = accessor<T>::get(trg);
    bool* mask_ptr = accessor<bool>::const_ptr(mask);
    trg_ptr[i] = mask_ptr[i] ? value : src_ptr[i];
  } 

  return trg;
}

template <typename T>
Tensor<uint32_t> argmax(const Tensor<T>& src, uint32_t axis){
  if(src.empty()){
    return Tensor<uint32_t>();
  }

  if(axis < 0){
    axis = src.ndim() + axis;
  }

  if(axis >= src.ndim()){
    throw std::invalid_argument("Axis out of range");
  }

  shape_t new_shape = src.shape();
  new_shape.erase(new_shape.begin() + axis);

  Tensor<uint32_t> trg = Tensor<uint32_t>(new_shape);

  #pragma omp parallel for
  for (int i = 0; i < src.size(); ++i) {
    uint32_t* trg_ptr = accessor<uint32_t>::get(trg);
    T* src_ptr = accessor<T>::const_ptr(src);
    uint32_t max_index = 0;
    T max_value = src_ptr[i];
    for(int j = 1; j < src.shape()[axis]; j++){
      if(src_ptr[i] > max_value){
        max_value = src_ptr[i];
        max_index = j;
      }
    }
    trg_ptr[i] = max_index;
  }
}
#endif