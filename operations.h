#ifndef OPERATIONS_H
#define OPERATIONS_H
#include <numeric>
#include <cmath>
#include "kernels.h"
#include "tensor.h"



int getIndex(const std::vector<uint32_t>& indices, const std::vector<uint32_t>& strides) {
    int index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return index;
}


template <typename T>
Tensor<T> softmax(Tensor<T>* t, int axis)
{ 
  if(t->empty() || t == nullptr){
    return Tensor<T>();
  }

  if(axis < 0){
    axis = t->shape().size() + axis;
  }

  Tensor<T> out = Tensor<T>(t->shape());


  int dimSize = t->shape()[axis];
  int innerStride = 1;
  for (int i = axis + 1; i < t->shape().size(); ++i) {
      innerStride *= t->shape()[i];
  }
  int outerStride = innerStride * dimSize;

  #pragma omp parallel for
  for (int outer = 0; outer < t->size() / outerStride; ++outer) {
      #pragma unroll
      for (int inner = 0; inner < innerStride; ++inner) {
          T* out_ptr = accessor<T>::get(out);
          T* t_ptr = accessor<T>::const_ptr(*t);
          // Find the maximum value for numerical stability
          T maxVal = -std::numeric_limits<double>::infinity();
          for (int i = 0; i < dimSize; ++i) {
              int index = outer * outerStride + i * innerStride + inner;
              maxVal = std::max(maxVal, t_ptr[index]); 
          }

          // Compute the sum of exponentials
          T sumExp = 0.0;
          for (int i = 0; i < dimSize; ++i) {
              int index = outer * outerStride + i * innerStride + inner;
              out_ptr[index] = std::exp(t_ptr[index] - maxVal); // Subtract maxVal for numerical stability
              sumExp += out_ptr[index];
          }

          // Normalize to get the softmax probabilities
          for (int i = 0; i < dimSize; ++i) {
              int index = outer * outerStride + i * innerStride + inner;
              out_ptr[index] /= sumExp;
          }
      }
  }

  return out; 
}


template <typename T>
Tensor<T> expand(Tensor<T>* src, shape_t new_shape)
{

  Tensor<T> trg = Tensor<T>(new_shape);
  uint32_t src_size = src->size();
  uint32_t trg_size = trg.size();

  uint32_t dim = src->shape().size();

  T* src_ptr = accessor<T>::const_ptr(*src);
  T* trg_ptr = accessor<T>::get(trg);

  #pragma omp parallel for
  for (int i = 0; i < trg_size; ++i) {
    int src_index = 0;
    for(int j = 0; j < dim; j++){
      int idx = (i / trg.stride()[j]) % src->shape()[j];
      src_index += idx * src->stride()[j];
    }
    trg_ptr[i] = src_ptr[src_index];
  } 

  return trg;
}


template <typename T>
Tensor<T> batch_matmul(const Tensor<T>* a, const Tensor<T>* b)
{
  if(a->empty() || b->empty()){
    return Tensor<T>();
  }

  if(b->ndim() > a->ndim()){
    assert(false && "The second tensor should have less or equal dimensions than the first tensor");
  }

  bool condition = false;
  //shape_t new_shape = shape_t();

  if(a->ndim() == b->ndim() && a->ndim() > 1){

    condition = a->shape()[a->ndim() - 1] != b->shape()[b->ndim() - 2];
    assert(!condition && "Matrix multiplication is not compatible");
    // take the first n - 2 dimensions of a and the last dimension of b
    shape_t new_shape = shape_t();

    for(int i = 0; i < a->ndim() - 1; i++){
      new_shape.push_back(a->shape()[i]);
    }
    new_shape.push_back(b->shape()[b->ndim() - 1]);

    uint32_t batch_size = std::accumulate(new_shape.begin(), 
                                          new_shape.end() - 2, 
                                          1, std::multiplies<uint32_t>());

    Tensor<T> out = Tensor<T>(new_shape);

    uint32_t M = a->shape()[a->ndim() - 2];
    uint32_t N = b->shape()[b->ndim() - 1];
    uint32_t K = a->shape()[a->ndim() - 1];

    #pragma omp parallel for
    for(int i = 0; i < batch_size; i++){
      gemm<T>(accessor<T>::const_ptr(*a) + i * M * K, 
              accessor<T>::const_ptr(*b) + i * K * N, 
              accessor<T>::get(out) + i * M * N, 
              M, N, K);
    }

    return out;

  }else if(b->ndim() == 1 && a->ndim() > 1){
    // if b is 1 dimensional, then we are doing a dot product
    // check if the last dimension of two tensors are the same
    condition = a->shape()[a->ndim() - 1] != b->shape()[0];
    assert(!condition && "Matrix multiplication is not compatible");

    shape_t new_shape = shape_t();
    for(int i = 0; i < a->ndim() - 1; i++){
      new_shape.push_back(a->shape()[i]);
    }

    Tensor<T> out = Tensor<T>(new_shape);
    uint32_t batch_size = std::accumulate(new_shape.begin(), 
                                          new_shape.end(), 
                                          1, std::multiplies<uint32_t>());
    
    uint32_t ddot_size = a->shape()[a->ndim() - 1];
    
    #pragma omp parallel for
    for(int i = 0; i < batch_size; i++){
      dot<T>(accessor<T>::const_ptr(*a) + i * ddot_size, 
            accessor<T>::const_ptr(*b), 
            accessor<T>::get(out) + i, 
            ddot_size);
    } 

    return out;
  }
  
  // in other case, we have A ndim > B ndim and B ndim > 1
  uint32_t M = a->shape()[a->ndim() - 2];
  uint32_t N = b->shape()[b->ndim() - 1];
  uint32_t K = a->shape()[a->ndim() - 1];

  shape_t new_shape = shape_t();
  for(int i = 0; i < a->ndim() - 1; i++){
    new_shape.push_back(a->shape()[i]);
  }
  new_shape.push_back(b->shape()[b->ndim() - 1]);
  Tensor<T> out = Tensor<T>(new_shape);
  uint32_t batch_size = std::accumulate(new_shape.begin(), 
                                        new_shape.end() - 2, 
                                        1, std::multiplies<uint32_t>());

  #pragma omp parallel for
  for(int i = 0; i < batch_size; i++){
    gemm<T>(accessor<T>::const_ptr(*a) + i * M * K, 
            accessor<T>::const_ptr(*b), 
            accessor<T>::get(out) + i * M * N, 
            M, N, K);
  }

  return out;
}


// simple case: A has higher dimension then B,
// B has only 1 dimension
template <typename T>
Tensor<T> broadcast_add(const Tensor<T>* a, const Tensor<T>* b){
  Tensor<T> out = Tensor<T>(a->shape());
  shape_t new_shape = a->shape();
  uint32_t batch_size = std::accumulate(new_shape.begin(), 
                                        new_shape.end() - 1, 
                                        1, std::multiplies<uint32_t>()); 

  uint32_t slices = a->size() / batch_size;

  #pragma omp parallel for
  for(int i = 0; i < batch_size; i++){
    #pragma unroll
    for(int j = 0; j < slices; j++){
      T* out_ptr = accessor<T>::get(out) + i * slices + j;
      T* a_ptr = accessor<T>::const_ptr(*a) + i * slices + j;
      T* b_ptr = accessor<T>::const_ptr(*b);
      *out_ptr = *a_ptr + *b_ptr;
    } 
  } 
  return out;
}
#endif