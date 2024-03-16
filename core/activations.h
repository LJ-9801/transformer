#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <numeric>
#include "tensor.h"


template <typename T, bool Inplace>
using InplaceType = typename std::enable_if<Inplace, void>::type;

template <typename T, bool Inplace>
using NonInplaceType = typename std::enable_if<!Inplace, Tensor<T>>::type;

template <typename T, bool Inplace>
InplaceType<T, Inplace> ReLU(Tensor<T>& t)
{
  if(t.empty()){
    return;
  }

  #pragma omp parallel for
  for (int i = 0; i < t.size(); ++i) {
      T* t_ptr = accessor<T>::get(t);
      t_ptr[i] = std::max(t_ptr[i], (T)0);
  }
}

template <typename T, bool Inplace>
NonInplaceType<T, Inplace> ReLU(const Tensor<T>& t)
{
  if(t.empty()){
    return Tensor<T>();
  }

  Tensor<T> out = Tensor<T>(t.shape());
  #pragma omp parallel for
  for (int i = 0; i < t.size(); ++i) {
      T* out_ptr = accessor<T>::get(out);
      out_ptr[i] = std::max(t[i], (T)0);
  }
  return out;
}

template <typename T, bool Inplace>
InplaceType<T, Inplace> dropout(Tensor<T>& t, float p = 0.5)
{
  if(t.empty()){
    return;
  }

  #pragma omp parallel for
  for (int i = 0; i < t.size(); ++i) {
      T* t_ptr = accessor<T>::get(t);
      t_ptr[i] = (rand() / (RAND_MAX + 1.0)) < p ? 0 : t_ptr[i];
  }
}

template <typename T, bool Inplace>
NonInplaceType<T, Inplace> dropout(const Tensor<T>& t, float p = 0.5)
{
  if(t.empty()){
    return Tensor<T>();
  }

  Tensor<T> out = Tensor<T>(t.shape());
  #pragma omp parallel for
  for (int i = 0; i < t.size(); ++i) {
      T* out_ptr = accessor<T>::get(out);
      out_ptr[i] = (rand() / (RAND_MAX + 1.0)) < p ? 0 : t[i];
  }
  return out;
}


template <typename T>
Tensor<T> softmax(const Tensor<T>& t, int axis)
{ 
  if(t.empty()){
    return Tensor<T>();
  }

  if(axis < 0){
    axis = t.ndim() + axis;
  }

  Tensor<T> out = Tensor<T>(t.shape());

  int dimSize = t.shape()[axis];
  int innerStride = 1;
  for (int i = axis + 1; i < t.ndim(); ++i) {
      innerStride *= t.shape()[i];
  }
  int outerStride = innerStride * dimSize;

  #pragma omp parallel for
  for (int outer = 0; outer < t.size() / outerStride; ++outer) {
      #pragma unroll
      for (int inner = 0; inner < innerStride; ++inner) {
          T* out_ptr = accessor<T>::get(out);
          T* t_ptr = accessor<T>::const_ptr(t);
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


#endif