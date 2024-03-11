#include "multiheadattention.h"



int main(){

  Tensor<float> src = Tensor<float>({3, 1, 4});
  src.arange();

  for (int i = 0; i < src.size(); ++i) {
    cout << src.data()[i] << " ";
  }
  cout << endl;

  Tensor<float> trg_exp = expand(&src, {3, 4, 4});

  return 0;
}