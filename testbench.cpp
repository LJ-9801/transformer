#include "multiheadattention.h"



int main(){

  Tensor<float> key = Tensor<float>({3, 2, 4});
  Tensor<float> value = Tensor<float>({3, 2, 4});
  Tensor<float> query = Tensor<float>({3, 2, 4});

  key.fill_one();
  value.fill_one();
  query.fill_one();

  Multiheadattention mha(4, 2);

  mha.generate_weights();

  Tensor<float> out = mha.forward(key, query, value);

  for(int i = 0; i < out.size(); i++){
    cout << out.data()[i] << " ";
  }
  cout << endl;

  return 0;
}