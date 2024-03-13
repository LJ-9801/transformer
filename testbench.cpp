#include "multiheadattention.h"
#include "layernorm.h"



int main(){

  auto key = Tensor<float>({4, 1, 6}).arange();
  auto value = Tensor<float>({4, 1, 6}).arange();
  auto query = Tensor<float>({4, 1, 6}).arange();

  Multiheadattention mha(6, 2);

  mha.generate_weights();

  auto out = mha.forward(key, query, value);

  LayerNorm norm({1 , 6});
  auto out2 = norm.forward(key);

  for(int i = 0; i < out.size(); i++){
    std::cout << out2[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}