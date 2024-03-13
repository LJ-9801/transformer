#include "multiheadattention.h"
#include "core/layernorm.h"



int main(){

  auto key = Tensor<float>({4, 1, 6}).arange();
  auto value = Tensor<float>({4, 1, 6}).arange();
  auto query = Tensor<float>({4, 1, 6}).arange();

  Multiheadattention mha(6, 2);
  LayerNorm norm({1 , 6});
  Linear lin(6, 6);

  mha.generate_weights();
  lin.generate_weights();
  
  auto out = mha.forward(key, query, value);
  auto out2 = norm.forward(key);
  auto out3 = lin.forward(key);

  return 0;
}