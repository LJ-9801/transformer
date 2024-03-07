#pragma once
#ifndef KERNELS_H


// @todo: needs to be optimized
template <typename T>
void gemm(const T* a, const T* b, const T* c, T* d, int M, int N, int K)
{
  #pragma omp parallel for
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      T sum = 0;
      #pragma unroll
      for(int k = 0; k < K; k++){
        sum += a[i * K + k] * b[k * N + j];
      }
      if(c != nullptr)
        d[i * N + j] = sum + c[i * N + j];
      else
        d[i * N + j] = sum;
    }
  }
}
#endif