#include "multiheadattention.h"



int main(){

  Tensor<float> key = Tensor<float>({512, 8, 64});
  Tensor<float> value = Tensor<float>({512, 8, 64});
  Tensor<float> query = Tensor<float>({512, 8, 64});

  key.fill_one();
  value.fill_one();
  query.fill_one();

  Multiheadattention mha(64, 8);

  mha.generate_weights();

  Tensor<float> out = mha.forward(key, query, value);

  return 0;
}