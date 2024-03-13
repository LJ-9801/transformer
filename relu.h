#ifndef RELU_H
#define RELU_H
#include "kernels.h"
#include "tensor.h"

class ReLU{
  public:
  ReLU(bool inplace = false):_inplace(inplace) {}

  Tensor<float> forward(Tensor<float> input){

  }

  private:
  bool _inplace;

};
#endif