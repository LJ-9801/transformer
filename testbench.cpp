#include "multiheadattention.h"



int main(){

  auto key = Tensor<float>({4, 1, 6}).arange();
  auto value = Tensor<float>({4, 1, 6}).arange();
  auto query = Tensor<float>({4, 1, 6}).arange();

  Multiheadattention mha(6, 2);

  mha.generate_weights();

  auto out = mha.forward(key, query, value);
  
  for(int i = 0; i < out.size(); i++){
    std::cout << out[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}