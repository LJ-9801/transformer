#include "multiheadattention.h"



int main(){
  float* a_data = new float[16]{
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  float* b_data = new float[4]{
    1, 2, 3, 4
  };

  Tensor<float> a = Tensor<float>(
        a_data,
        {2, 2, 2, 2});

  Tensor<float> b = Tensor<float>(
        b_data, 
        {2, 2});

  Tensor<float> c = batch_matmul<float>(&a, &b, nullptr);

  c.transpose({1, 2});

  return 0;
}