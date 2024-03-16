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
#endif