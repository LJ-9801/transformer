#ifndef OPERATIONS_H
#define OPERATIONS_H
#include <numeric>
#include <cmath>
#include "kernels.h"
#include "tensor.h"

/*
  here we are just doing a simple batch matmul
  where tensor A has to be bigger then tensor B

  the last two dimension of A and B has to follow
  the matmul rule
*/
template <typename T>
Tensor<T> batch_matmul(const Tensor<T>* a, const Tensor<T>* b, const Tensor<T>* c)
{
  if(a->empty() || b->empty()){
    return Tensor<T>();
  }

  uint32_t M = a->shape()[a->shape().size() - 2];
  uint32_t Ka = a->shape()[a->shape().size() - 1];
  uint32_t Kb = b->shape()[b->shape().size() - 2];
  uint32_t N = b->shape()[b->shape().size() - 1];

  assert(Ka == Kb && "The last two dimensions of A and B must be the same");
  bool condition = (a->shape().size() == 4 && b->shape().size() == 2) || (a->shape().size() == 4 && b->shape().size() == 4);
  assert(condition && "The shape of A and B is not supported");
  // here we are assuming size A > size B, 
  // fow now let's just consider A has dim 4 and B has dim 2
  uint32_t total_size = a->size();
  uint32_t nBatch = a->shape()[0] * a->shape()[1];

  T* b_ptr = accessor<T>::const_ptr(*b);
  T* c_ptr = nullptr;
  if(c != nullptr){
    c_ptr = accessor<T>::const_ptr(*c);
  }

  Tensor<T> out = Tensor<T>({a->shape()[0], a->shape()[1], M, N});
  T* out_ptr = accessor<T>::get(out);


  // we need to handle both a is 4D and b is 2D and also a is 4D and b is also 4D
  #pragma omp parallel for
  for(int i = 0; i < nBatch; i++){
    T* a_ptr = accessor<T>::const_ptr(*a) + i * M * Ka;
    if(b->shape().size() == 4){
      b_ptr = accessor<T>::const_ptr(*b) + i * N * Kb;
    }
    T* out_ptr = accessor<T>::get(out) + i * M * N;
    gemm(a_ptr, b_ptr, c_ptr, out_ptr, M, N, Ka);
  }

  return out;
}

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
#endif