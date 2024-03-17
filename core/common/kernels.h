#pragma once
#ifndef KERNELS_H
#define KERNELS_H
#include <iostream>
#include <cmath>
#include <cblas.h>
#include "utilities.h"

#define TILE_M 128
#define TILE_N 64
#define TILE_K 64

#define multiply_accumulate(begin, end) \
      std::accumulate(begin, end, 1, std::multiplies<uint32_t>())

// @todo: needs to be optimized
template <typename T>
void gemm(const T* a, const T* b, T* d, int M, int N, int K)
{
  #pragma omp parallel for collapse(2)
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      T sum = 0;
      #pragma unroll
      for(int k = 0; k < K; k++){
        sum += a[i * K + k] * b[k * N + j];
      }
      d[i * N + j] = sum; 
    }
  }
}


// @todo: needs to be optimized
// the most important kernel as it is used all
// over transformer
template <typename T>
void gemm_nt(const T* a, const T* b, const T *c, T* d, int M, int N, int K)
{

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i += TILE_M) {
    for (int j = 0; j < N; j += TILE_N) {
      for (int k = 0; k < K; k += TILE_K) {
        int M2 = std::min(i + TILE_M, M);
        int N2 = std::min(j + TILE_N, N);
        int K2 = std::min(k + TILE_K, K);
        for (int i2 = i; i2 < M2; i2++) {
          for (int j2 = j; j2 < N2; j2++) {
            T sum = 0;
            #pragma unroll
            for (int k2 = k; k2 < K2; k2++) {
              sum += a[i2 * K + k2] * b[j2 * K + k2];
            }
            d[i2 * N + j2] += c == nullptr ? sum : sum + c[i2 * N + j2];
          }
        }
      }
    }
  }

  //openblas_set_num_threads(4);
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, a, K, b, K, 0.0, d, N);

  /*
  //#pragma omp parallel for collapse(2) schedule(dynamic)
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      T sum = 0;
      #pragma unroll
      for(int k = 0; k < K; k++){
        sum += a[i * K + k] * b[j * K + k];
      }
      d[i * N + j] = sum; 
    }
  }
  */
}


// this function is used to expand a tensor
// for example there is a tensor with shape [32, 1, 512]
// and we want to expand it to [32, 10, 512]
template <typename T>
void expand_kernel(T* src, T* trg,
            uint32_t* src_shape, uint32_t* trg_shape, 
            uint32_t* src_stride, uint32_t* trg_stride, 
            uint32_t src_size, uint32_t trg_size,
            uint32_t dim){
  
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < trg_size; ++i) {
    uint32_t indices[dim];
    uint32_t index = i;
    for (int j = dim - 1; j >= 0; --j) {
      indices[j] = index / trg_stride[j];
      index = index % trg_stride[j];
    }
    int src_index = index_from_stride(indices, src_stride, dim);
    trg[i] = src[src_index]; 
  }
}


template <typename T>
void dot(const T* in1, const T* in2, T *output, const size_t size){
  *output = 0;
  #pragma unroll
  for(int i = 0; i < size; i++){
    *output += in1[i] * in2[i];
  } 
}

template <typename T>
void tranpose_tensor(const T* input, T** output, 
                    const uint32_t *old_shape, const uint32_t *new_shape,
                    const uint32_t *old_stride, const uint32_t *new_stride,
                    const size_t size, const uint32_t dim,
                    const uint16_t axis1, const uint16_t axis2){
  
  #pragma omp parallel for
  for(size_t i = 0; i<size; i++){
    uint32_t old_pos[dim];
    for(uint32_t j = 0; j < dim; j++){
      old_pos[j] = (i / old_stride[j]) % old_shape[j];
    }

    std::swap(old_pos[axis1], old_pos[axis2]);

    uint32_t new_index = 0;
    for(uint32_t j = 0; j < dim; j++){
      new_index += old_pos[j] * new_stride[j];
    }
    (*output)[new_index] = input[i];
  }

}
#endif