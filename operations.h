#ifndef OPERATIONS_H
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
  assert(a->shape().size() == 4 && b->shape().size() == 2 && "A has to be a 4D tensor and B has to be a 2D tensor");

  // here we are assuming size A > size B, 
  // fow now let's just consider A has dim 4 and B has dim 2
  uint32_t total_size = std::accumulate(a->shape().begin(), a->shape().end(), 1, std::multiplies<uint32_t>());
  uint32_t nBatch = a->shape()[0] * a->shape()[1];

  T* b_ptr = accessor<T>::const_ptr(*b);
  T* c_ptr = nullptr;
  if(c != nullptr){
    c_ptr = accessor<T>::const_ptr(*c);
  }

  Tensor<T> out = Tensor<T>({a->shape()[0], a->shape()[1], M, N});
  T* out_ptr = accessor<T>::get(out);

  #pragma omp parallel for
  for(int i = 0; i < nBatch; i++){
    T* a_ptr = accessor<T>::const_ptr(*a) + i * M * Ka;
    T* out_ptr = accessor<T>::get(out) + i * M * N;
    gemm(a_ptr, b_ptr, c_ptr, out_ptr, M, N, Ka);
  }

  return out;
}

template <typename T>
Tensor<T> softmax(Tensor<T>* t, int axis)
{ 
  if(t->empty() || t == nullptr){
    return Tensor<T>();
  }


  return Tensor<T>();
}
#endif