#include "multiheadattention.h"



int main(){

  Tensor<float> src = Tensor<float>({3, 4, 4});
  src.arange();

  Tensor<float> trg_exp = Tensor<float>({4, 4});
  trg_exp.arange();

  Tensor<float> out = batch_matmul<float>(&src, &trg_exp, nullptr);

  for(int i = 0; i < out.size(); i++){
    std::cout << out.data()[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}