#include "multiheadattention.h"



int main(){

  Tensor<float> src = Tensor<float>({3, 1, 4});
  src.arange();

  for (int i = 0; i < src.size(); ++i) {
    cout << src.data()[i] << " ";
  }
  cout << endl;

  Tensor<float> trg_exp = expand(&src, {3, 4, 4});

  for (int i = 0; i < trg_exp.size(); ++i) {
    cout << trg_exp.data()[i] << " ";
  }

  return 0;
}