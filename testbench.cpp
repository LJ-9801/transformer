#include "multiheadattention.h"



int main(){
  Multiheadattention mha(512, 8);
  mha.generate_weights();

  auto key = Tensor<float>(vector<float>(1.0, 512 * 10), {1, 10, 512});
  auto query = Tensor<float>(vector<float>(1.0, 512 * 10), {1, 10, 512});
  auto value = Tensor<float>(vector<float>(1.0, 512 * 10), {1, 10, 512});

  auto output = mha.forward(key, query, value);

  return 0;
}