#ifndef DROPOUT_H
#define DROPOUT_H
#include "kernels.h"
#include "tensor.h"

class Dropout{
  public:
  
  Dropout(float p = 0.5, bool inplace = false):
          _p(p), _inplace(inplace) {}

  Tensor<float> forward(Tensor<float> input){

  }

  private:

  float _p;
  bool _inplace;

};

#endif